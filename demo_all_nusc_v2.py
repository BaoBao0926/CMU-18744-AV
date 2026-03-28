"""
NuScenes CAM_FRONT demo: DeepLabV3 semantic seg + YOLOPv2 drivable/lane + route planning.
"""
import os
import heapq
import cv2
import numpy as np
import torch
from scipy import ndimage
import albumentations as A
import yaml
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from YOLOPv2.utils.utils import (
    letterbox,
    select_device,
    scale_coords,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    plot_one_box,
)
import DeeplabV3.misc.segm.lookup_table as lut

# --- Config (globals) ---
DATAROOT = "/home/zl3466/Documents/dataset/NuScenes"
SCENE_NAME = "scene-0945"
DISPLAY_SIZE = (1280, 720)
OUTPUT_ROOT = os.path.join("output", "NuScenes")

IMGSZ, STRIDE = 640, 32
YOLO_WEIGHTS = "YOLOPv2/weights/yolopv2.pt"
CONF_THRES, IOU_THRES = 0.3, 0.45

DEEPLAB_DIR = os.path.join("DeeplabV3", "weights", "sem_segm_gps_split")
DEEPLAB_MODEL_PATH = os.path.join(DEEPLAB_DIR, "DeeplabV3Plus_EfficientNetB4_best_model_epoch_0060_workzone.pth")
DEEPLAB_CONFIG_PATH = os.path.join(DEEPLAB_DIR, "DeeplabV3Plus_EfficientNetB4_workzone.yaml")

FREE, OBSTACLE = 0, 1
CELL_SIZE = 16
SAFETY_DISTANCE_PIXELS = 60
CENTER_BIAS, CENTER_CAP = 0.2, 4


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_colormap(device):
    classes = [
        (0, 255, (0, 0, 0)), (1, 1, (70, 70, 70)), (2, 2, (102, 102, 156)),
        (3, 3, (190, 153, 153)), (4, 4, (180, 165, 180)), (5, 5, (150, 100, 100)),
        (6, 6, (246, 116, 185)), (7, 7, (248, 135, 182)), (8, 8, (251, 172, 187)),
        (9, 9, (255, 68, 51)), (10, 10, (255, 104, 66)), (11, 11, (184, 107, 35)),
        (12, 12, (205, 135, 29)), (13, 13, (30, 119, 179)), (14, 14, (44, 79, 206)),
        (15, 15, (102, 81, 210)), (16, 16, (170, 118, 213)), (17, 17, (214, 154, 219)),
        (18, 18, (241, 71, 14)), (19, 19, (254, 139, 32)), (0, 0, (0, 0, 0)),
    ]
    l_key = [[c[1]] for c in classes]
    l_color = [c[2] for c in classes]
    _, th_lut = lut.get_lookup_table(
        ar_u_key=np.asarray(l_key, dtype=np.uint8),
        ar_u_val=np.asarray(l_color, dtype=np.uint8),
        v_val_default=0, device=device,
    )
    return th_lut


# ---------- YOLOPv2 ----------
def apply_yolo_seg_overlay(image_bgr, da_seg_mask, ll_seg_mask):
    if da_seg_mask is not None:
        m = da_seg_mask == 1
        image_bgr[m] = (image_bgr[m] * 0.1 + np.array([0, 255, 0]) * 0.9).astype(image_bgr.dtype)
    if ll_seg_mask is not None:
        m = ll_seg_mask == 1
        image_bgr[m] = (image_bgr[m] * 0.1 + np.array([0, 0, 255]) * 0.9).astype(image_bgr.dtype)


def run_yolopv2(yolo_model, device, im0_bgr, half):
    h, w = im0_bgr.shape[:2]
    img, _, _ = letterbox(im0_bgr, IMGSZ, stride=STRIDE)
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    img = torch.from_numpy(img).to(device)
    img = (img.half() if half else img.float()) / 255.0
    img = img.unsqueeze(0)
    with torch.no_grad():
        [pred, anchor_grid], seg, ll = yolo_model(img)
    pred = split_for_trace_model(pred, anchor_grid)
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)
    da = driving_area_mask(seg)
    ll_mask = lane_line_mask(ll)
    if da.shape != (h, w):
        da = cv2.resize(da.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if ll_mask.shape != (h, w):
        ll_mask = cv2.resize(ll_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    det = pred[0] if pred and len(pred) else torch.zeros((0, 6), device=device)
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_bgr.shape).round()
    return det, da, ll_mask


# ---------- DeepLabV3 ----------
def run_deeplab(deeplab_model, device, image_bgr, transform_full, transform_padded, th_lut):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform_full(image=image_rgb)
    with torch.inference_mode():
        x = torch.from_numpy(transformed["image"]).unsqueeze(0).to(device)
        logits = deeplab_model(x)
        pred = logits.argmax(axis=1)
    img_pad = transform_padded(image=image_rgb)["image"]
    pred_color = lut.lookup_chw(td_u_input=pred.byte(), td_i_lut=th_lut).permute(1, 2, 0).cpu().numpy()
    pred_ids = pred.squeeze(0).cpu().numpy()
    valid = np.setdiff1d(np.arange(1, 20), [1])
    mask = np.isin(pred_ids, valid)
    blend = img_pad.copy()
    if mask.any():
        blend[mask] = cv2.addWeighted(img_pad[mask], 0.1, pred_color[mask], 0.9, 0.0)
    return cv2.cvtColor(blend, cv2.COLOR_RGB2BGR), pred_ids


def _compute_grid_dims(h, w, base_cells=20):
    short_side = min(h, w)
    long_side = max(h, w)
    long_cells = int(round(base_cells * long_side / float(short_side)))
    long_cells = max(base_cells, long_cells)
    if h <= w:
        return base_cells, long_cells
    return long_cells, base_cells


def _resize_mask_nearest(mask, target_shape):
    th, tw = target_shape
    if mask.shape == (th, tw):
        return mask
    return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)


def compute_high_level_semantics(pred_ids, drivable_mask, base_image_bgr=None):
    if base_image_bgr is not None:
        bh, bw = base_image_bgr.shape[:2]
        if pred_ids.shape != (bh, bw):
            pred_ids = cv2.resize(pred_ids.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
        drivable_mask = _resize_mask_nearest(drivable_mask, (bh, bw))
    h, w = pred_ids.shape
    drivable_mask = _resize_mask_nearest(drivable_mask, (h, w))

    workzone_ids = np.array([6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.uint8)
    road_ids = np.array([1, 3], dtype=np.uint8)
    sidewalk_id = 2
    vehicle_ids = np.array([9, 10], dtype=np.uint8)

    rows, cols = _compute_grid_dims(h, w)
    y_edges = np.linspace(0, h, rows + 1).astype(int)
    x_edges = np.linspace(0, w, cols + 1).astype(int)

    labels = np.empty((rows, cols), dtype=object)

    workzone_th = 0.10
    non_drivable_vehicle_th = 0.08
    non_drivable_road_th = 0.30
    non_drivable_drive_max = 0.25
    openlane_drivable_th = 0.40
    openlane_road_th = 0.30
    openlane_workzone_max = 0.05
    openlane_vehicle_max = 0.05
    sidewalk_th = 0.25

    for r in range(rows):
        y0, y1 = y_edges[r], y_edges[r + 1]
        for c in range(cols):
            x0, x1 = x_edges[c], x_edges[c + 1]
            if y1 <= y0 or x1 <= x0:
                labels[r, c] = "non-drivable area"
                continue
            cell_pred = pred_ids[y0:y1, x0:x1]
            cell_drive = drivable_mask[y0:y1, x0:x1]
            area = float(cell_pred.size)
            if area == 0:
                labels[r, c] = "non-drivable area"
                continue
            workzone_ratio = np.isin(cell_pred, workzone_ids).sum() / area
            road_ratio = np.isin(cell_pred, road_ids).sum() / area
            sidewalk_ratio = (cell_pred == sidewalk_id).sum() / area
            drivable_ratio = (cell_drive == 1).sum() / area
            vehicle_ratio = np.isin(cell_pred, vehicle_ids).sum() / area

            if workzone_ratio >= workzone_th:
                labels[r, c] = "workzone"
            elif vehicle_ratio >= non_drivable_vehicle_th or (road_ratio >= non_drivable_road_th and drivable_ratio <= non_drivable_drive_max):
                labels[r, c] = "non-drivable area"
            elif drivable_ratio >= openlane_drivable_th and road_ratio >= openlane_road_th and workzone_ratio < openlane_workzone_max and vehicle_ratio < openlane_vehicle_max:
                labels[r, c] = "open lane"
            elif sidewalk_ratio >= sidewalk_th:
                labels[r, c] = "sidewalk area"
            else:
                labels[r, c] = "non-drivable area"

    visited = np.zeros((rows, cols), dtype=bool)
    regions = []
    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue
            label = labels[r, c]
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            min_r = max_r = r
            min_c = max_c = c
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and labels[nr, nc] == label:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            regions.append({
                "label": label,
                "cells": cells,
                "bbox": (min_r, max_r, min_c, max_c),
            })

    return labels, regions, (x_edges, y_edges)


def draw_semantic_regions(base_image_bgr, regions, x_edges, y_edges, fill_alpha=0.18):
    viz = base_image_bgr.copy()
    label_colors = {
        "workzone": (0, 140, 255),
        "open lane": (0, 200, 0),
        "sidewalk area": (0, 215, 255),
        "non-drivable area": (0, 0, 200),
    }
    for region in regions:
        label = region["label"]
        min_r, max_r, min_c, max_c = region["bbox"]
        x0, x1 = x_edges[min_c], x_edges[max_c + 1]
        y0, y1 = y_edges[min_r], y_edges[max_r + 1]
        if y1 <= y0 or x1 <= x0:
            continue
        color = label_colors.get(label, (0, 0, 0))
        if fill_alpha >= 1.0:
            cv2.rectangle(viz, (x0, y0), (x1 - 1, y1 - 1), color, -1)
        else:
            overlay = viz.copy()
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)
            viz = cv2.addWeighted(overlay, fill_alpha, viz, 1.0 - fill_alpha, 0)
        cv2.rectangle(viz, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
        tx = x0 + 6
        ty = y0 + 18
        cv2.putText(viz, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(viz, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return viz


# ---------- Route planning ----------
def _astar(grid, start, goal, dt=None):
    H, W = grid.shape
    sx, sy, gx, gy = *start, *goal
    if grid[sy, sx] != FREE or grid[gy, gx] != FREE:
        return []
    open_set = [(0, (sx, sy))]
    came_from, g_score = {}, {(sx, sy): 0}
    while open_set:
        _, (x, y) = heapq.heappop(open_set)
        if (x, y) == (gx, gy):
            path = []
            while (x, y) in came_from:
                path.append((x, y))
                x, y = came_from[(x, y)]
            path.append((sx, sy))
            return path[::-1]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == FREE:
                    cost = 1.414 if dx and dy else 1.0
                    if dt is not None:
                        cost = max(0.05, cost - CENTER_BIAS * min(dt[ny, nx], CENTER_CAP))
                    ng = g_score[(x, y)] + cost
                    if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = ng
                        h = ((nx - gx) ** 2 + (ny - gy) ** 2) ** 0.5
                        heapq.heappush(open_set, (ng + h, (nx, ny)))
                        came_from[(nx, ny)] = (x, y)
    return []


def _smooth_polyline(path, iterations=5):
    if len(path) <= 2:
        return path
    pts = np.array(path, dtype=np.float64)
    for _ in range(iterations):
        pts_new = pts.copy()
        for i in range(1, len(pts) - 1):
            pts_new[i] = 0.5 * (pts[i - 1] + pts[i + 1])
        pts = pts_new
    return [(int(round(x)), int(round(y))) for x, y in pts]


def _cell_center(cx, cy, cs):
    return (cx * cs + cs // 2, cy * cs + cs // 2)


def plan_route(drivable_mask, safety_px=40):
    H, W = drivable_mask.shape
    binary = (drivable_mask == 1).astype(np.uint8)
    cs = CELL_SIZE
    Hc, Wc = H // cs, W // cs
    if Hc < 1 or Wc < 1:
        return [], binary, np.full_like(binary, OBSTACLE), None, None
    coarse = binary[: Hc * cs, : Wc * cs].reshape(Hc, cs, Wc, cs).any(axis=(1, 3)).astype(np.uint8)
    safety_c = max(1, safety_px // cs)
    big = float(Hc + Wc)
    d_left = d_right = d_top = np.full((Hc, Wc), big, dtype=np.float64)
    xv, yv = np.arange(Wc, dtype=np.float64), np.arange(Hc, dtype=np.float64)
    for y in range(Hc):
        row = coarse[y, :]
        z = np.flatnonzero(row == 0)
        if len(z):
            i = np.searchsorted(z, xv, side="right") - 1
            d_left[y] = np.where(row == 0, 0, np.where(i >= 0, xv - z[i], big))
            i = np.minimum(np.searchsorted(z, xv, side="left"), len(z) - 1)
            d_right[y] = np.where(row == 0, 0, z[i] - xv)
    for x in range(Wc):
        col = coarse[:, x]
        z = np.flatnonzero(col == 0)
        if len(z):
            i = np.searchsorted(z, yv, side="right") - 1
            d_top[:, x] = np.where(col == 0, 0, np.where(i >= 0, yv - z[i], big))
    dist = np.minimum(np.minimum(d_left, d_right), d_top)
    shrunk = ((coarse > 0) & (dist >= safety_c)).astype(np.uint8)
    labeled, n = ndimage.label(shrunk)
    if n == 0:
        return [], binary, np.full_like(binary, OBSTACLE), None, None
    best = max(range(1, n + 1), key=lambda i: np.sum(labeled == i))
    comp = labeled == best
    ys, xs = np.where(comp)
    min_y = np.min(ys)
    top = ys == min_y
    goal_c = (int(np.round(np.mean(xs[top]))), int(min_y))
    grid = np.where(comp, FREE, OBSTACLE).astype(np.uint8)
    grid_pixel = np.kron(grid, np.ones((cs, cs), dtype=grid.dtype))[:H, :W]
    dt = ndimage.distance_transform_edt(grid == FREE).astype(np.float32)
    mid_c = (Wc // 2, Hc - 1)
    fy, fx = np.where(grid == FREE)
    if len(fy) == 0:
        return [], binary, grid_pixel, None, _cell_center(goal_c[0], goal_c[1], cs)
    i = np.argmin((fx - mid_c[0]) ** 2 + (fy - mid_c[1]) ** 2)
    start_c = (int(fx[i]), int(fy[i]))
    path_c = _astar(grid, start_c, goal_c, dt=dt)
    if not path_c:
        return [], binary, grid_pixel, _cell_center(start_c[0], start_c[1], cs), _cell_center(goal_c[0], goal_c[1], cs)
    path_px = [_cell_center(cx, cy, cs) for (cx, cy) in path_c]
    return _smooth_polyline(path_px), binary, grid_pixel, _cell_center(start_c[0], start_c[1], cs), _cell_center(goal_c[0], goal_c[1], cs)


def draw_route_overlay(im, path_pixels, binary, grid, start_pixel, goal_pixel):
    overlay = np.zeros_like(im)
    overlay[binary == 1] = [0, 180, 0]
    overlay[grid == FREE] = [200, 100, 0]
    im[:] = cv2.addWeighted(im, 0.65, overlay, 0.35, 0)
    contours, _ = cv2.findContours((grid == FREE).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if c.size >= 4:
            cv2.polylines(im, [c], True, (255, 255, 0), 2, cv2.LINE_AA)
    if start_pixel:
        cv2.circle(im, start_pixel, 10, (0, 255, 0), 2)
    if goal_pixel:
        cv2.circle(im, goal_pixel, 10, (0, 0, 255), 2)
    if len(path_pixels) >= 2:
        cv2.polylines(im, [np.array(path_pixels, dtype=np.int32)], False, (0, 255, 255), 3, cv2.LINE_AA)


def draw_legend(im):
    deeplab_items = [
        (1, "Road", (70, 70, 70)), (2, "Sidewalk", (156, 102, 102)), (3, "Bike Lane", (153, 153, 190)),
        (4, "Off-Road", (180, 165, 180)), (5, "Roadside", (100, 100, 150)),
        (6, "Barrier", (185, 116, 246)), (7, "Barricade", (182, 135, 248)), (8, "Fence", (187, 172, 251)),
        (9, "Police Vehicle", (51, 68, 255)), (10, "Work Vehicle", (66, 104, 255)),
        (11, "Police Officer", (35, 107, 184)), (12, "Worker", (29, 135, 205)),
        (13, "Cone", (179, 119, 30)), (14, "Drum", (206, 79, 44)), (15, "Vertical Panel", (210, 81, 102)),
        (16, "Tubular Marker", (213, 118, 170)), (17, "Work Equipment", (219, 154, 214)),
        (18, "Arrow Board", (14, 71, 241)), (19, "TTC Sign", (32, 139, 254)),
    ]
    yolo_items = [("Driving Area", (0, 255, 0)), ("Lane Line", (0, 0, 255)), ("Route", (0, 255, 255))]
    items = [(name, color) for _, name, color in deeplab_items] + yolo_items
    font, scale, pad, swatch, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 10, 12, 18
    legend_w = 240
    h, w = im.shape[:2]
    x0, y0 = max(w - legend_w - pad, pad), pad
    x1, y1 = x0 + legend_w, y0 + pad * 2 + line_h * len(items)
    cv2.rectangle(im, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 0), 1)
    y = y0 + pad + line_h - 4
    for name, color in items:
        cv2.rectangle(im, (x0 + pad, y - swatch + 2), (x0 + pad + swatch, y + 2), color, -1)
        cv2.putText(im, name, (x0 + pad + swatch + 6, y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_h


# ---------- Main ----------
if __name__ == "__main__":
    device = select_device("0")
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=True)
    scene = next(s for s in nusc.scene if s["name"] == SCENE_NAME)
    sd_token = nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"]
    paths = []
    while sd_token:
        sd = nusc.get("sample_data", sd_token)
        paths.append(os.path.join(DATAROOT, sd["filename"]))
        sd_token = sd["next"]

    yolo = torch.jit.load(YOLO_WEIGHTS).to(device).eval()
    half = device.type != "cpu"
    if half:
        yolo.half()
    if device.type != "cpu":
        yolo(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).type_as(next(yolo.parameters())))

    config = yaml.safe_load(open(DEEPLAB_CONFIG_PATH))
    preprocess = get_preprocessing_fn(config["segmentation_model_backbone"], config["segmentation_pretrained_dataset"])
    transform_full = A.Compose([
        A.Resize(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]),
        A.Lambda(name="preprocess", image=preprocess),
        A.PadIfNeeded(736, 1280),
        A.Lambda(name="to_tensor", image=to_tensor),
    ])
    transform_padded = A.Compose([A.Resize(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]), A.PadIfNeeded(736, 1280)])
    deeplab = torch.load(DEEPLAB_MODEL_PATH, map_location=device, weights_only=False)
    deeplab.eval()
    th_lut = get_colormap(device)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out_base = os.path.join(OUTPUT_ROOT, f"{SCENE_NAME}_all.mp4")
    out_grid = os.path.join(OUTPUT_ROOT, f"{SCENE_NAME}_all_grid.mp4")
    out_grid_only = os.path.join(OUTPUT_ROOT, f"{SCENE_NAME}_grid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_base, fourcc, 10, DISPLAY_SIZE)
    writer_grid = cv2.VideoWriter(out_grid, fourcc, 10, DISPLAY_SIZE)
    writer_grid_only = cv2.VideoWriter(out_grid_only, fourcc, 10, DISPLAY_SIZE)

    for path in tqdm(paths, desc="CAM_FRONT"):
        im0 = cv2.resize(cv2.imread(path), DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        det, da, ll = run_yolopv2(yolo, device, im0, half)
        deeplab_vis, pred_ids = run_deeplab(deeplab, device, im0, transform_full, transform_padded, th_lut)
        merged = cv2.resize(deeplab_vis, DISPLAY_SIZE)
        if len(det):
            for k in range(len(det) - 1, -1, -1):
                plot_one_box(det[k, :4].cpu().tolist(), merged, line_thickness=3)
        apply_yolo_seg_overlay(merged, da, ll)
        path_px, binary, grid, start_px, goal_px = plan_route(da, safety_px=SAFETY_DISTANCE_PIXELS)
        draw_route_overlay(merged, path_px, binary, grid, start_px, goal_px)
        draw_legend(merged)

        _, regions, (x_edges, y_edges) = compute_high_level_semantics(pred_ids, da, base_image_bgr=merged)
        merged_with_grid = draw_semantic_regions(merged, regions, x_edges, y_edges, fill_alpha=0.5)
        grid_only = draw_semantic_regions(np.zeros_like(merged), regions, x_edges, y_edges, fill_alpha=1.0)

        writer.write(merged)
        writer_grid.write(merged_with_grid)
        writer_grid_only.write(grid_only)

    writer.release()
    writer_grid.release()
    writer_grid_only.release()
