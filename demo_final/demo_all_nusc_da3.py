"""
NuScenes CAM_FRONT demo: DeepLabV3 semantic seg + YOLOPv2 drivable/lane + route planning, with workzone mask
"""
import os
import heapq
import io
import contextlib
import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree
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
from depth_anything_3.api import DepthAnything3


# --- Config (globals) ---
DATAROOT = "/home/zl3466/Documents/dataset/NuScenes"
SCENE_NAME = "scene-0945"
# SCENE_NAME = "scene-0956"
DISPLAY_SIZE = (1280, 720)
OUTPUT_VIS_DIR = f"output/overlay_depth/{SCENE_NAME}"

IMGSZ, STRIDE = 640, 32
YOLO_WEIGHTS = "YOLOPv2/weights/yolopv2.pt"
CONF_THRES, IOU_THRES = 0.3, 0.45

DEEPLAB_DIR = os.path.join("DeeplabV3", "weights", "sem_segm_gps_split")
DEEPLAB_MODEL_PATH = os.path.join(DEEPLAB_DIR, "DeeplabV3Plus_EfficientNetB4_best_model_epoch_0060_workzone.pth")
DEEPLAB_CONFIG_PATH = os.path.join(DEEPLAB_DIR, "DeeplabV3Plus_EfficientNetB4_workzone.yaml")

# barrier, barricade, fence, cone, drum, work equipment (DeepLab class IDs)
WORKZONE_CLASSES = [6, 7, 8, 13, 14, 17]
WORKZONE_CONF_THRESH = 0.7
WORKZONE_CLUSTER_DIST_M = 8  # merge workzone instances within this 3D distance (meters)
WORKZONE_MIN_INSTANCE_AREA_PX = 300  # drop tiny connected components before clustering
WORKZONE_MIN_CONF_MASS = WORKZONE_MIN_INSTANCE_AREA_PX * WORKZONE_CONF_THRESH  # min sum of softmax confidence over instance pixels
WORKZONE_CLOSE_KERNEL = 31  # odd; morphological close to bridge gaps
CLEARANCE_KDTREE_MAX_POINTS = 10000  # cap surface points (non-drivable 3D) for k-d tree

FREE, OBSTACLE = 0, 1
CELL_SIZE = 16  # coarse grid for route + 3D clearance sampling
SAFETY_DISTANCE_METERS = 0.5
CENTER_BIAS, CENTER_CAP = 0.2, 4

# Smooth YOLOP drivable ∪ lane: drop tiny interior undrivable specks before workzone subtract.
# Kernel size is scaled by min(H,W) vs 720p reference.
DRIVABLE_CLOSE_KERNEL_REF = 9  # odd reference kernel at min display dimension 720


# Convert HWC image to CHW float tensor
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


# Build color lookup table for DeepLab class IDs
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


def scale_intrinsics(K, orig_size, new_size):
    """Scale camera intrinsics from orig_size (w,h) to new_size (w,h)."""
    ow, oh = orig_size
    nw, nh = new_size
    sx = float(nw) / float(ow)
    sy = float(nh) / float(oh)
    K2 = K.astype(np.float32).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return K2


def depth_to_xyz_map(depth_m, K):
    """Unproject a depth map (meters) to per-pixel XYZ in camera frame."""
    H, W = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth_m.astype(np.float32, copy=False)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def extract_workzone_instances(pred_ids, conf_map, xyz_map):
    """Return list of instances with centroid pixel + centroid XYZ."""
    h, w = xyz_map.shape[:2]
    evidence = np.isin(pred_ids, WORKZONE_CLASSES) & (conf_map >= float(WORKZONE_CONF_THRESH))
    lab, n = ndimage.label(evidence.astype(np.uint8))
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        if int(comp.sum()) < int(WORKZONE_MIN_INSTANCE_AREA_PX):
            continue
        if float(conf_map[comp].sum()) < float(WORKZONE_MIN_CONF_MASS):
            continue
        ys, xs = np.where(comp)
        cy = int(np.clip(np.round(ys.mean()), 0, h - 1))
        cx = int(np.clip(np.round(xs.mean()), 0, w - 1))
        p = xyz_map[cy, cx]
        if not np.isfinite(p).all() or p[2] <= 0:
            continue
        out.append({"mask": comp, "centroid_px": (cx, cy), "centroid_xyz": p})
    return out


def _fill_cluster_region(mask_bool, h, w):
    """Turn a possibly disconnected cluster mask into one connected region (convex hull of pixels)."""
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w), dtype=np.uint8)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    if pts.shape[0] == 1:
        out[int(ys[0]), int(xs[0])] = 1
        return out
    if pts.shape[0] == 2:
        cv2.line(out, (int(xs[0]), int(ys[0])), (int(xs[1]), int(ys[1])), 1, thickness=3)
        return out
    try:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices].astype(np.int32)
        cv2.fillConvexPoly(out, poly, 1)
    except Exception:
        out[ys, xs] = 1
    return out


def compute_workzone_mask_clustered(im0_bgr, pred_ids, conf_map, xyz_map):
    """Cluster workzone instances by 3D distance, then join each cluster into one connected mask.

    Returns:
        merged_px: uint8 mask after clustering, hull fill, and morphological close.
        comps: list from extract_workzone_instances (same frame); reuse for debug / metrics.
    """
    h, w = im0_bgr.shape[:2]
    pred_ids = pred_ids[:h, :w]
    conf_map = conf_map[:h, :w]

    comps = extract_workzone_instances(pred_ids, conf_map, xyz_map)
    m = len(comps)
    if m == 0:
        return np.zeros((h, w), dtype=np.uint8), []

    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    thr = float(WORKZONE_CLUSTER_DIST_M)
    for i in range(m):
        for j in range(i + 1, m):
            d = float(np.linalg.norm(comps[i]["centroid_xyz"] - comps[j]["centroid_xyz"]))
            if d <= thr:
                union(i, j)

    clusters = {}
    for i in range(m):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    merged_px = np.zeros((h, w), dtype=np.uint8)
    for idxs in clusters.values():
        merged = np.zeros((h, w), dtype=bool)
        for k in idxs:
            merged |= comps[k]["mask"]
        filled = _fill_cluster_region(merged, h, w)
        merged_px = np.maximum(merged_px, filled)

    k = int(WORKZONE_CLOSE_KERNEL)
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    merged_px = cv2.morphologyEx(merged_px, cv2.MORPH_CLOSE, kernel, iterations=1)
    return merged_px, comps


# ---------- YOLOPv2 ----------
# Overlay YOLOP driving area and lane line masks on an image
def apply_yolo_seg_overlay(image_bgr, da_seg_mask, ll_seg_mask):
    if da_seg_mask is not None:
        m = da_seg_mask == 1
        image_bgr[m] = (image_bgr[m] * 0.1 + np.array([0, 255, 0]) * 0.9).astype(image_bgr.dtype)
    if ll_seg_mask is not None:
        m = ll_seg_mask == 1
        image_bgr[m] = (image_bgr[m] * 0.1 + np.array([0, 0, 255]) * 0.9).astype(image_bgr.dtype)


# Run YOLOPv2 to get detections, drivable area mask, and lane line mask
def run_yolopv2(yolo_model, device, im0_bgr):
    h, w = im0_bgr.shape[:2]
    img, _, _ = letterbox(im0_bgr, IMGSZ, stride=STRIDE)
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    img = torch.from_numpy(img).to(device)
    # img = (img.half() if half else img.float()) / 255.0
    img = img.float() / 255.0
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


def smooth_drivable_union(da_seg, ll_seg, ref_kernel=DRIVABLE_CLOSE_KERNEL_REF):
    """drivable ∪ lane lines as uint8 {0,1}; close + fill holes to remove small gaps inside drivable."""
    raw = ((da_seg == 1) | (ll_seg == 1)).astype(np.uint8)
    h, w = raw.shape
    k = max(3, int(round(float(ref_kernel) * min(h, w) / 720.0)))
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, ker, iterations=1)
    filled = ndimage.binary_fill_holes(closed.astype(bool))
    return filled.astype(np.uint8)


# ---------- DeepLabV3 ----------
# Run DeepLab model and return colorized segmentation and class ID map
def run_deeplab(deeplab_model, device, image_bgr, transform_full, transform_padded, th_lut):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform_full(image=image_rgb)
    with torch.inference_mode():
        x = torch.from_numpy(transformed["image"]).unsqueeze(0).to(device)
        logits = deeplab_model(x)
        probs = torch.softmax(logits, dim=1)
        conf = probs.max(dim=1).values
        pred = logits.argmax(axis=1)
    img_pad = transform_padded(image=image_rgb)["image"]
    pred_color = lut.lookup_chw(td_u_input=pred.byte(), td_i_lut=th_lut).permute(1, 2, 0).cpu().numpy()
    pred_ids = pred.squeeze(0).cpu().numpy()
    conf_map = conf.squeeze(0).float().cpu().numpy()
    valid = np.setdiff1d(np.arange(1, 20), [1])
    mask = np.isin(pred_ids, valid)
    blend = img_pad.copy()
    if mask.any():
        blend[mask] = cv2.addWeighted(img_pad[mask], 0.1, pred_color[mask], 0.9, 0.0)
    return cv2.cvtColor(blend, cv2.COLOR_RGB2BGR), pred_ids, conf_map


# (Workzone mask is computed via 3D centroid clustering; see `compute_workzone_mask_clustered`.)
# ---------- Route planning ----------
# A* search on a coarse grid with optional distance transform bias
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


# Simple iterative smoothing of a polyline path
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


# Convert grid cell indices to pixel center coordinates
def _cell_center(cx, cy, cs):
    return (cx * cs + cs // 2, cy * cs + cs // 2)


# Plan a safe route on the drivable mask using a coarse occupancy grid
def plan_route(drivable_mask, depth_m, safety_m=5.0, xyz_map=None):
    H, W = drivable_mask.shape
    if depth_m.shape != drivable_mask.shape:
        depth_m = cv2.resize(depth_m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    depth_m = depth_m.astype(np.float32, copy=False)
    finite_depth = np.isfinite(depth_m) & (depth_m > 0)
    cs = int(CELL_SIZE)

    # Single 3D clearance: nearest distance to any non-drivable surface point (workzone, curb, etc.).
    # drivable_mask already excludes workzone interior; those pixels contribute to the obstacle cloud.
    keep_clear = np.ones((H, W), dtype=bool)
    if xyz_map is not None:
        if xyz_map.shape[:2] != (H, W):
            xyz_map = np.stack(
                [
                    cv2.resize(xyz_map[..., i], (W, H), interpolation=cv2.INTER_NEAREST)
                    for i in range(3)
                ],
                axis=-1,
            ).astype(np.float32)
        surf = (drivable_mask != 1) & finite_depth
        surf_pts = xyz_map[surf].reshape(-1, 3)
        ok = np.isfinite(surf_pts).all(axis=1) & (surf_pts[:, 2] > 0)
        surf_pts = surf_pts[ok]
        n_s = surf_pts.shape[0]
        cap = int(CLEARANCE_KDTREE_MAX_POINTS)
        if n_s > cap:
            surf_pts = surf_pts[np.linspace(0, n_s - 1, cap, dtype=int)]
        if n_s > 0:
            tree = cKDTree(surf_pts)
            off = [0, cs // 2, max(0, cs - 1)]
            Q_list, metas = [], []
            for y0 in range(0, H, cs):
                y1 = min(y0 + cs, H)
                for x0 in range(0, W, cs):
                    x1 = min(x0 + cs, W)
                    qs = []
                    for oy in off:
                        for ox in off:
                            yy = min(y0 + min(oy, y1 - y0 - 1), y1 - 1)
                            xx = min(x0 + min(ox, x1 - x0 - 1), x1 - 1)
                            if drivable_mask[yy, xx] == 1 and finite_depth[yy, xx]:
                                qs.append(xyz_map[yy, xx])
                    if not qs:
                        continue
                    Q_list.append(np.asarray(qs, dtype=np.float64))
                    metas.append((y0, y1, x0, x1, len(qs)))
            if Q_list:
                Q = np.vstack(Q_list)
                dist, _ = tree.query(Q, k=1)
                i0 = 0
                thr = float(safety_m)
                for (y0, y1, x0, x1, nq) in metas:
                    if dist[i0 : i0 + nq].min() < thr:
                        keep_clear[y0:y1, x0:x1] = False
                    i0 += nq

    binary = ((drivable_mask == 1) & finite_depth & keep_clear).astype(np.uint8)
    Hc, Wc = H // cs, W // cs
    if Hc < 1 or Wc < 1:
        return [], binary, np.full_like(binary, OBSTACLE), None, None
    mid_c = (Wc // 2, Hc - 1)  # bottom-middle cell in coarse grid coordinates
    # Light-green mask is derived on a coarse grid:
    # - Downsample `binary` into `coarse` where each CELL_SIZE×CELL_SIZE block is free if ANY pixel is free.
    # - Prefer the connected component that contains the bottom-middle cell; otherwise fall back to largest.
    coarse = binary[: Hc * cs, : Wc * cs].reshape(Hc, cs, Wc, cs).any(axis=(1, 3)).astype(np.uint8)
    labeled, n = ndimage.label(coarse)
    if n == 0:
        return [], binary, np.full_like(binary, OBSTACLE), None, None
    preferred = int(labeled[mid_c[1], mid_c[0]])
    if preferred != 0:
        comp = labeled == preferred
    else:
        best = max(range(1, n + 1), key=lambda i: np.sum(labeled == i))
        comp = labeled == best
    ys, xs = np.where(comp)
    min_y = np.min(ys)
    top = ys == min_y
    # Pick a goal that is guaranteed to lie on a FREE cell (avoid rounding into gaps).
    xs_top = np.sort(xs[top])
    goal_c = (int(xs_top[len(xs_top) // 2]), int(min_y))
    grid = np.where(comp, FREE, OBSTACLE).astype(np.uint8)
    # Upsample the coarse FREE/OBSTACLE grid back to image pixels for visualization (this is the light-green area).
    grid_pixel = np.kron(grid, np.ones((cs, cs), dtype=grid.dtype))[:H, :W]
    dt = ndimage.distance_transform_edt(grid == FREE).astype(np.float32)
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


# Draw drivable area, free grid cells, and the planned route on the image
def draw_route_overlay(im, path_pixels, binary, grid, start_pixel, goal_pixel):
    overlay = np.zeros_like(im)
    overlay[binary == 1] = [0, 180, 0]
    overlay[grid == FREE] = [200, 100, 0]
    im[:] = cv2.addWeighted(im, 0.65, overlay, 0.35, 0)
    # Draw the outer boundary of the light-green free space:
    # contours are extracted from the `grid == FREE` mask and rendered as closed polylines.
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


# Draw a legend showing DeepLab classes, YOLOP masks, and route color
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
# Load NuScenes frames, run YOLOP + DeepLab, plan route, and visualize workzones
if __name__ == "__main__":
    # Select device and load NuScenes scene metadata
    device = select_device("0")
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=True)
    scene = next(s for s in nusc.scene if s["name"] == SCENE_NAME)
    sd_token = nusc.get("sample", scene["first_sample_token"])["data"]["CAM_FRONT"]
    frames = []
    while sd_token:
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.asarray(cs["camera_intrinsic"], dtype=np.float32)
        orig_size = (int(sd["width"]), int(sd["height"]))
        frames.append(
            {
                "path": os.path.join(DATAROOT, sd["filename"]),
                "K": scale_intrinsics(K, orig_size=orig_size, new_size=DISPLAY_SIZE),
            }
        )
        sd_token = sd["next"]

    # Preprocessing
    config = yaml.safe_load(open(DEEPLAB_CONFIG_PATH))
    preprocess = get_preprocessing_fn(config["segmentation_model_backbone"], config["segmentation_pretrained_dataset"])
    transform_full = A.Compose([
        A.Resize(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]),
        A.Lambda(name="preprocess", image=preprocess),
        A.PadIfNeeded(736, 1280),
        A.Lambda(name="to_tensor", image=to_tensor),
    ])
    transform_padded = A.Compose([A.Resize(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]), A.PadIfNeeded(736, 1280)])

    # Load YOLOPV2 model (optionally half-precision on GPU)
    yolo = torch.jit.load(YOLO_WEIGHTS).to(device).eval()
    if device.type != "cpu":
        yolo(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).type_as(next(yolo.parameters())))

    # Load DeepLabV3 model
    deeplab = torch.load(DEEPLAB_MODEL_PATH, map_location=device, weights_only=False)
    deeplab.eval()
    th_lut = get_colormap(device)

    # Load DepthAnything3 model
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3Metric-Large")
    da3 = da3.to(device=device)


    # Process each CAM_FRONT frame: detect, segment, plan route, and visualize
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    for frame_idx, frame in enumerate(tqdm(frames, desc="CAM_FRONT")):
        im0 = cv2.resize(cv2.imread(frame["path"]), DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        # run yolopv2, get vehicle bbox, drivable area, and lane line mask
        det, da, ll = run_yolopv2(yolo, device, im0)
        # run deeplab, get semantic segmentation mask
        deeplab_vis, pred_ids, conf_map = run_deeplab(deeplab, device, im0, transform_full, transform_padded, th_lut)
        # run depthanything3, get metric depth map (pixel value in meters)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            depth = da3.inference([im0]).depth[0]
        depth_m = cv2.resize(depth.astype(np.float32), DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
        xyz_map = depth_to_xyz_map(depth_m, frame["K"])
        workzone_mask, wz_instances = compute_workzone_mask_clustered(im0, pred_ids, conf_map, xyz_map)
        wz_bool = workzone_mask.astype(bool)

        # format depth map for visualization
        depth_vis = (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min()) * 255
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_vis = cv2.resize(depth_vis, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
        
        merged = cv2.resize(deeplab_vis, DISPLAY_SIZE)
        if len(det):
            for k in range(len(det) - 1, -1, -1):
                plot_one_box(det[k, :4].cpu().tolist(), merged, line_thickness=3)
        apply_yolo_seg_overlay(merged, da, ll)

        # Drivable for planning: smoothed YOLOP drivable ∪ lane lines, minus workzone pixels
        drivable_union = smooth_drivable_union(da, ll)
        drivable_for_plan = (drivable_union.astype(bool) & ~wz_bool).astype(np.uint8)
        path_px, binary, grid, start_px, goal_px = plan_route(
            drivable_for_plan,
            depth_m,
            safety_m=SAFETY_DISTANCE_METERS,
            xyz_map=xyz_map,
        )
        draw_route_overlay(merged, path_px, binary, grid, start_px, goal_px)
        draw_legend(merged)

        if wz_instances:
            for i, inst in enumerate(wz_instances):
                cx, cy = inst["centroid_px"]
                x, y, z = inst["centroid_xyz"]
                print(f"[frame {frame_idx:06d}] workzone_obj {i}: centroid_px=({cx},{cy}) centroid_xyz=({x:.2f},{y:.2f},{z:.2f})")
            for i in range(len(wz_instances)):
                for j in range(i + 1, len(wz_instances)):
                    d = float(np.linalg.norm(wz_instances[i]["centroid_xyz"] - wz_instances[j]["centroid_xyz"]))
                    print(f"[frame {frame_idx:06d}] dist obj{i}-obj{j} = {d:.2f} m")
        mask = wz_bool
        if mask.any():
            merged[mask] = (merged[mask] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(merged.dtype)
            num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if area == 0:
                    continue
                label = "workzone area"
                text_x = max(0, x + 2)
                text_y = max(12, y + 14)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(merged, (text_x - 2, text_y - th - 4), (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)
                cv2.putText(merged, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Show overlay result and depth map side-by-side
        vis = np.hstack([merged, depth_vis])
        cv2.imwrite(os.path.join(OUTPUT_VIS_DIR, f"{frame_idx:06d}.jpg"), vis)
        cv2.imshow("Overlay | Depth", vis)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
