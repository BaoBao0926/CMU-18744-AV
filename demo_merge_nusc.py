import os
import cv2
import numpy as np
import torch
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

# NuScenes
DATAROOT = "/home/zl3466/Documents/dataset/NuScenes"
SCENE_NAME = "scene-0945"

# YOLOPv2
IMGSZ = 640
STRIDE = 32
YOLO_WEIGHTS = "YOLOPv2/weights/yolopv2.pt"
CONF_THRES = 0.3
IOU_THRES = 0.45

# DeepLabV3
DEEPLAB_DIR = os.path.join("DeeplabV3", "weights", "sem_segm_gps_split")
DEEPLAB_MODEL_PATH = os.path.join(DEEPLAB_DIR, "DeeplabV3Plus_EfficientNetB4_best_model_epoch_0060_workzone.pth")
DEEPLAB_CONFIG_PATH = os.path.join(DEEPLAB_DIR, "DeeplabV3Plus_EfficientNetB4_workzone.yaml")

DISPLAY_SIZE = (1280, 720)


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
    l_key_trainid = [[c[1]] for c in classes]
    l_key_color = [c[2] for c in classes]
    ar_u_key_trainid = np.asarray(l_key_trainid, dtype=np.uint8)
    ar_u_key_color = np.asarray(l_key_color, dtype=np.uint8)
    _, th_i_lut_trainid2color = lut.get_lookup_table(
        ar_u_key=ar_u_key_trainid, ar_u_val=ar_u_key_color, v_val_default=0, device=device,
    )
    return th_i_lut_trainid2color


def apply_yolo_seg_overlay(image_bgr, da_seg_mask, ll_seg_mask):
    if da_seg_mask is not None:
        mask = da_seg_mask == 1
        image_bgr[mask] = (image_bgr[mask] * 0.1 + np.array([0, 255, 0]) * 0.9).astype(image_bgr.dtype)
    if ll_seg_mask is not None:
        mask = ll_seg_mask == 1
        image_bgr[mask] = (image_bgr[mask] * 0.1 + np.array([0, 0, 255]) * 0.9).astype(image_bgr.dtype)


def run_yolopv2_frame(yolo_model, device, im0_bgr, half):
    """Run YOLOPv2 on one frame (BGR, HxW). Returns det (Nx6), da_seg_mask, ll_seg_mask at im0 shape."""
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
    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    if da_seg_mask.shape != (h, w):
        da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if ll_seg_mask.shape != (h, w):
        ll_seg_mask = cv2.resize(ll_seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    det = pred[0] if pred and len(pred) else torch.zeros((0, 6), device=device)
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_bgr.shape).round()
    return det, da_seg_mask, ll_seg_mask


def run_deeplabv3(deeplab_model, device, image_bgr, transform_full, transform_padded, th_i_lut_trainid2color):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform_full(image=image_rgb)
    img_tensor = transformed["image"]
    with torch.inference_mode():
        model_input = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
        logits = deeplab_model(model_input)
        prediction = logits.argmax(axis=1)
    img_padded = transform_padded(image=image_rgb)["image"]
    prediction_color = lut.lookup_chw(
        td_u_input=prediction.byte(), td_i_lut=th_i_lut_trainid2color,
    ).permute((1, 2, 0)).cpu().numpy()
    pred_ids = prediction.squeeze(0).cpu().numpy()
    valid_ids = np.arange(1, 20, dtype=pred_ids.dtype)
    valid_ids = valid_ids[valid_ids != 1]
    mask = np.isin(pred_ids, valid_ids)
    blend = img_padded.copy()
    blend[mask] = cv2.addWeighted(img_padded[mask], 0.1, prediction_color[mask], 0.9, 0.0)
    blend_bgr = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
    return blend_bgr


def draw_legend(image_bgr):
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
    yolo_items = [("Driving Area", (0, 255, 0)), ("Lane Line", (0, 0, 255))]
    items = [(name, color) for _, name, color in deeplab_items] + yolo_items
    font, scale, pad, swatch, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 10, 12, 18
    legend_h, legend_w = pad * 2 + line_h * len(items), 240
    h, w = image_bgr.shape[:2]
    x0, y0 = max(w - legend_w - pad, pad), pad
    x1, y1 = x0 + legend_w, y0 + legend_h
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 0, 0), 1)
    y = y0 + pad + line_h - 4
    for name, color in items:
        cv2.rectangle(image_bgr, (x0 + pad, y - swatch + 2), (x0 + pad + swatch, y + 2), color, -1)
        cv2.putText(image_bgr, name, (x0 + pad + swatch + 6, y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_h
    return image_bgr


# --- Main ---
device = select_device("0")
nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=True)
scene = next(s for s in nusc.scene if s["name"] == SCENE_NAME)
first_sample = nusc.get("sample", scene["first_sample_token"])
sd_token = first_sample["data"]["CAM_FRONT"]
paths = []
while sd_token:
    cam_sd = nusc.get("sample_data", sd_token)
    paths.append(os.path.join(DATAROOT, cam_sd["filename"]))
    sd_token = cam_sd["next"]

# YOLOPv2
yolo_model = torch.jit.load(YOLO_WEIGHTS).to(device).eval()
half = device.type != "cpu"
if half:
    yolo_model.half()
if device.type != "cpu":
    yolo_model(torch.zeros(1, 3, IMGSZ, IMGSZ).to(device).type_as(next(yolo_model.parameters())))

# DeepLabV3
config = yaml.safe_load(open(DEEPLAB_CONFIG_PATH, "r"))
preprocess_input = get_preprocessing_fn(
    config["segmentation_model_backbone"],
    config["segmentation_pretrained_dataset"],
)
transform_full = A.Compose([
    A.Resize(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]),
    A.Lambda(name="image_preprocessing", image=preprocess_input),
    A.PadIfNeeded(736, 1280),
    A.Lambda(name="to_tensor", image=to_tensor),
])
transform_padded = A.Compose([
    A.Resize(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]),
    A.PadIfNeeded(736, 1280),
])
deeplab_model = torch.load(DEEPLAB_MODEL_PATH, map_location=device, weights_only=False)
deeplab_model.eval()
th_i_lut = get_colormap(device)

for path in tqdm(paths, desc="CAM_FRONT"):
    im0 = cv2.imread(path)
    im0 = cv2.resize(im0, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
    det, da_seg_mask, ll_seg_mask = run_yolopv2_frame(yolo_model, device, im0, half)
    deeplab_vis = run_deeplabv3(deeplab_model, device, im0, transform_full, transform_padded, th_i_lut)
    merged = cv2.resize(deeplab_vis, (DISPLAY_SIZE[0], DISPLAY_SIZE[1]))
    if det is not None and len(det):
        for k in range(len(det) - 1, -1, -1):
            xyxy = det[k, :4].cpu().tolist()
            plot_one_box(xyxy, merged, line_thickness=3)
    apply_yolo_seg_overlay(merged, da_seg_mask, ll_seg_mask)
    draw_legend(merged)
    cv2.imshow("DeepLabV3 + YOLOPv2", merged)
    cv2.waitKey(30)
cv2.destroyAllWindows()
