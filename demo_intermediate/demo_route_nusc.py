import os
import heapq
import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from YOLOPv2.utils.utils import (
    letterbox,
    driving_area_mask,
    lane_line_mask,
    show_seg_result,
    select_device,
)

# NuScenes + YOLOPv2
DATAROOT = "/home/zl3466/Documents/dataset/NuScenes"
SCENE_NAME = "scene-0945"
IMGSZ = 640
STRIDE = 32
WEIGHTS = "YOLOPv2/weights/yolopv2.pt"

# Route planning
FREE, OBSTACLE = 0, 1
CELL_SIZE = 16
SAFETY_DISTANCE_PIXELS = 60
CENTER_BIAS = 0.2  # prefer cells farther from boundary (0 = no bias, higher = stick to center)
CENTER_CAP = 4  # max distance (cells) that reduces cost


def _astar(grid, start, goal, dt=None):
    """
    A* on 2D grid (8-neighbors). start/goal (x, y).
    If dt (distance transform to boundary) is given, step cost is lower in the center of the corridor.
    Returns path as list of (x,y) or [].
    """
    H, W = grid.shape
    sx, sy = start
    gx, gy = goal
    if grid[sy, sx] != FREE or grid[gy, gx] != FREE:
        return []
    open_set = [(0, (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0}
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
                        d = min(dt[ny, nx], CENTER_CAP)
                        cost = max(0.05, cost - CENTER_BIAS * d)
                    ng = g_score[(x, y)] + cost
                    if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = ng
                        h = ((nx - gx) ** 2 + (ny - gy) ** 2) ** 0.5
                        heapq.heappush(open_set, (ng + h, (nx, ny)))
                        came_from[(nx, ny)] = (x, y)
    return []


def _smooth_polyline(path, iterations=5):
    """Smooth path with Laplacian smoothing; keep start/end fixed."""
    if len(path) <= 2:
        return path
    pts = np.array(path, dtype=np.float64)
    n = len(pts)
    for _ in range(iterations):
        pts_new = pts.copy()
        for i in range(1, n - 1):
            pts_new[i] = 0.5 * (pts[i - 1] + pts[i + 1])
        pts = pts_new
    return [(int(round(x)), int(round(y))) for x, y in pts]


def _cell_center(cx, cy, cs):
    return (cx * cs + cs // 2, cy * cs + cs // 2)


def plan_route(drivable_mask, depth_image=None, safety_distance_pixels=40):
    """
    Plan path from bottom-center to goal in drivable area.
    drivable_mask: (H, W) binary (1 = drivable). Returns path_pixels, binary, grid_pixel, start_pixel, goal_pixel.
    """
    H, W = drivable_mask.shape
    binary = (drivable_mask == 1).astype(np.uint8)
    cs = CELL_SIZE
    Hc, Wc = H // cs, W // cs
    if Hc < 1 or Wc < 1:
        return [], binary, np.full_like(binary, OBSTACLE), None, None
    coarse_binary = (
        binary[: Hc * cs, : Wc * cs]
        .reshape(Hc, cs, Wc, cs)
        .any(axis=(1, 3))
        .astype(np.uint8)
    )
    safety_cells = max(1, safety_distance_pixels // cs)
    big = float(Hc + Wc)
    d_left = np.full((Hc, Wc), big, dtype=np.float64)
    d_right = np.full((Hc, Wc), big, dtype=np.float64)
    d_top = np.full((Hc, Wc), big, dtype=np.float64)
    x_vals = np.arange(Wc, dtype=np.float64)
    y_vals = np.arange(Hc, dtype=np.float64)
    for y in range(Hc):
        row = coarse_binary[y, :]
        zeros = np.flatnonzero(row == 0)
        if len(zeros):
            idx = np.searchsorted(zeros, x_vals, side="right") - 1
            d_left[y, :] = np.where(row == 0, 0, np.where(idx >= 0, x_vals - zeros[idx], big))
            idx = np.searchsorted(zeros, x_vals, side="left")
            idx_safe = np.minimum(idx, len(zeros) - 1)
            d_right[y, :] = np.where(row == 0, 0, np.where(idx < len(zeros), zeros[idx_safe] - x_vals, big))
        else:
            d_left[y, :] = d_right[y, :] = big
    for x in range(Wc):
        col = coarse_binary[:, x]
        zeros = np.flatnonzero(col == 0)
        if len(zeros):
            idx = np.searchsorted(zeros, y_vals, side="right") - 1
            d_top[:, x] = np.where(col == 0, 0, np.where(idx >= 0, y_vals - zeros[idx], big))
        else:
            d_top[:, x] = big
    dist_no_bottom = np.minimum(np.minimum(d_left, d_right), d_top)
    binary_shrunk = ((coarse_binary > 0) & (dist_no_bottom >= safety_cells)).astype(np.uint8)
    labeled, n_comp = ndimage.label(binary_shrunk)
    if n_comp == 0:
        return [], binary, np.full_like(binary, OBSTACLE), None, None
    best_component = None
    if depth_image is not None and np.isfinite(depth_image).any():
        best_depth = -np.inf
        for i in range(1, n_comp + 1):
            ys, xs = np.where(labeled == i)
            py = np.clip(ys * cs + cs // 2, 0, H - 1)
            px = np.clip(xs * cs + cs // 2, 0, W - 1)
            depths = depth_image[py, px]
            valid = np.isfinite(depths) & (depths > 0)
            if not np.any(valid):
                continue
            mean_depth = np.mean(depths[valid])
            if mean_depth > best_depth:
                best_depth = mean_depth
                best_component = i
    else:
        best_area = 0
        for i in range(1, n_comp + 1):
            area = np.sum(labeled == i)
            if area > best_area:
                best_area = area
                best_component = i
    if best_component is None:
        return [], binary, np.full_like(binary, OBSTACLE), None, None
    comp_mask = labeled == best_component
    ys, xs = np.where(comp_mask)
    if depth_image is not None and np.isfinite(depth_image).any():
        gx_c, gy_c = int(np.round(np.mean(xs))), int(np.round(np.mean(ys)))
    else:
        min_y = np.min(ys)
        top_mask = ys == min_y
        gx_c, gy_c = int(np.round(np.mean(xs[top_mask]))), int(min_y)
    goal_cell = (gx_c, gy_c)
    grid = np.where(comp_mask, FREE, OBSTACLE).astype(np.uint8)
    grid_pixel = np.kron(grid, np.ones((cs, cs), dtype=grid.dtype))[:H, :W]
    # Distance to boundary: higher = more central; use to bias path toward corridor center
    dt = ndimage.distance_transform_edt(grid == FREE).astype(np.float32)
    bottom_mid_c = (Wc // 2, Hc - 1)
    free_ys, free_xs = np.where(grid == FREE)
    if len(free_ys) == 0:
        return [], binary, grid_pixel, None, _cell_center(goal_cell[0], goal_cell[1], cs)
    idx = np.argmin((free_xs - bottom_mid_c[0]) ** 2 + (free_ys - bottom_mid_c[1]) ** 2)
    start_cell = (int(free_xs[idx]), int(free_ys[idx]))
    path_cells = _astar(grid, start_cell, goal_cell, dt=dt)
    if not path_cells:
        return [], binary, grid_pixel, _cell_center(start_cell[0], start_cell[1], cs), _cell_center(goal_cell[0], goal_cell[1], cs)
    path_pixels = [_cell_center(cx, cy, cs) for (cx, cy) in path_cells]
    return _smooth_polyline(path_pixels), binary, grid_pixel, _cell_center(start_cell[0], start_cell[1], cs), _cell_center(goal_cell[0], goal_cell[1], cs)

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=True)
scene = next(s for s in nusc.scene if s["name"] == SCENE_NAME)

# Collect all CAM_FRONT sample_data paths (keyframes + intermediate frames)
first_sample = nusc.get("sample", scene["first_sample_token"])
sd_token = first_sample["data"]["CAM_FRONT"]
paths = []
while sd_token:
    cam_sd = nusc.get("sample_data", sd_token)
    paths.append(os.path.join(DATAROOT, cam_sd["filename"]))
    sd_token = cam_sd["next"]

# Load YOLOPv2
device = select_device("0")
model = torch.jit.load(WEIGHTS).to(device).eval()
half = device.type != "cpu"
if half:
    model.half()
if device.type != "cpu":
    model(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).type_as(next(model.parameters())))

for path in tqdm(paths, desc="CAM_FRONT"):
    im0 = cv2.imread(path)
    im0 = cv2.resize(im0, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img, _, _ = letterbox(im0, IMGSZ, stride=STRIDE)
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    img = torch.from_numpy(img).to(device)
    img = (img.half() if half else img.float()) / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        [_, _], seg, ll = model(img)

    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    h, w = im0.shape[:2]
    if da_seg_mask.shape != (h, w):
        da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if ll_seg_mask.shape != (h, w):
        ll_seg_mask = cv2.resize(ll_seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
    path_pixels, binary, grid, start_pixel, goal_pixel = plan_route(
        da_seg_mask, depth_image=None, safety_distance_pixels=SAFETY_DISTANCE_PIXELS
    )
    overlay = np.zeros_like(im0)
    overlay[binary == 1] = [0, 180, 0]
    overlay[grid == FREE] = [200, 100, 0]
    im0 = cv2.addWeighted(im0, 0.65, overlay, 0.35, 0)
    # Drivable area boundary as polylines (contour of planned component)
    contours, _ = cv2.findContours(
        (grid == FREE).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        if c.size >= 4:
            cv2.polylines(im0, [c], True, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    if start_pixel is not None:
        cv2.circle(im0, start_pixel, 10, (0, 255, 0), 2)
    if goal_pixel is not None:
        cv2.circle(im0, goal_pixel, 10, (0, 0, 255), 2)
    if len(path_pixels) >= 2:
        pts = np.array(path_pixels, dtype=np.int32)
        cv2.polylines(im0, [pts], False, (0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow("YOLOPv2 + Route", im0)
    cv2.waitKey(30)
cv2.destroyAllWindows()
