import argparse
import glob
import os
from pathlib import Path
import re

import cv2
import numpy as np
import torch
import albumentations as A
import yaml
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from YOLOPv2.utils.utils import (
    time_synchronized,
    select_device,
    scale_coords,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    plot_one_box,
    LoadImages,
)
import DeeplabV3.misc.segm.lookup_table as lut

YOLO_CLASS_NAMES = [
    "person",
    "rider",
    "truck",
    "car",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]


def increment_exp_dir(project_dir, name, exist_ok):
    base_dir = Path(project_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / name

    if exist_ok or not target.exists():
        return target

    exp_pattern = re.compile(r"^exp(\d+)$")
    max_n = 0
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        if entry.name == "exp":
            max_n = max(max_n, 1)
            continue
        match = exp_pattern.match(entry.name)
        if match:
            max_n = max(max_n, int(match.group(1)))

    next_n = max_n + 1 if max_n else 1
    return base_dir / f"exp{next_n}"


def resolve_sources(source):
    if not source:
        raise ValueError("No input source provided.")

    if any(ch in source for ch in ["*", "?", "["]):
        files = sorted(glob.glob(source))
    elif os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "*.*")))
    else:
        files = [source]

    img_formats = {"bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"}
    vid_formats = {"mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"}

    images = [f for f in files if f.split(".")[-1].lower() in img_formats]
    videos = [f for f in files if f.split(".")[-1].lower() in vid_formats]

    if not images and not videos:
        raise ValueError(f"No images or videos found in {source}.")

    if videos:
        raise ValueError("Demo_merge.py currently supports images only.")

    return images


def get_colormap(device):
    classes = [
        (0, 255, (0, 0, 0)),
        (1, 1, (70, 70, 70)),
        (2, 2, (102, 102, 156)),
        (3, 3, (190, 153, 153)),
        (4, 4, (180, 165, 180)),
        (5, 5, (150, 100, 100)),
        (6, 6, (246, 116, 185)),
        (7, 7, (248, 135, 182)),
        (8, 8, (251, 172, 187)),
        (9, 9, (255, 68, 51)),
        (10, 10, (255, 104, 66)),
        (11, 11, (184, 107, 35)),
        (12, 12, (205, 135, 29)),
        (13, 13, (30, 119, 179)),
        (14, 14, (44, 79, 206)),
        (15, 15, (102, 81, 210)),
        (16, 16, (170, 118, 213)),
        (17, 17, (214, 154, 219)),
        (18, 18, (241, 71, 14)),
        (19, 19, (254, 139, 32)),
        (0, 0, (0, 0, 0)),
    ]

    l_key_trainid = [[c[1]] for c in classes]
    l_key_color = [c[2] for c in classes]

    ar_u_key_trainid = np.asarray(l_key_trainid, dtype=np.uint8)
    ar_u_key_color = np.asarray(l_key_color, dtype=np.uint8)

    _, th_i_lut_trainid2color = lut.get_lookup_table(
        ar_u_key=ar_u_key_trainid,
        ar_u_val=ar_u_key_color,
        v_val_default=0,
        device=device,
    )
    return th_i_lut_trainid2color


def get_legend_items():
    # DeeplabV3 classes (train_id, name, color in BGR for cv2)
    deeplab_items = [
        (1, "Road", (70, 70, 70)),
        (2, "Sidewalk", (156, 102, 102)),
        (3, "Bike Lane", (153, 153, 190)),
        (4, "Off-Road", (180, 165, 180)),
        (5, "Roadside", (100, 100, 150)),
        (6, "Barrier", (185, 116, 246)),
        (7, "Barricade", (182, 135, 248)),
        (8, "Fence", (187, 172, 251)),
        (9, "Police Vehicle", (51, 68, 255)),
        (10, "Work Vehicle", (66, 104, 255)),
        (11, "Police Officer", (35, 107, 184)),
        (12, "Worker", (29, 135, 205)),
        (13, "Cone", (179, 119, 30)),
        (14, "Drum", (206, 79, 44)),
        (15, "Vertical Panel", (210, 81, 102)),
        (16, "Tubular Marker", (213, 118, 170)),
        (17, "Work Equipment", (219, 154, 214)),
        (18, "Arrow Board", (14, 71, 241)),
        (19, "TTC Sign", (32, 139, 254)),
    ]

    yolo_items = [
        ("Driving Area", (0, 255, 0)),
        ("Lane Line", (0, 0, 255)),
    ]

    return deeplab_items, yolo_items


def draw_legend(image_bgr):
    deeplab_items, yolo_items = get_legend_items()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding = 10
    swatch = 12
    line_h = 18

    items = [(name, color) for _, name, color in deeplab_items] + yolo_items
    legend_h = padding * 2 + line_h * len(items)
    legend_w = 240

    h, w = image_bgr.shape[:2]
    x0 = max(w - legend_w - padding, padding)
    y0 = padding
    x1 = x0 + legend_w
    y1 = y0 + legend_h

    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)

    y = y0 + padding + line_h - 4
    for name, color in items:
        cv2.rectangle(
            image_bgr,
            (x0 + padding, y - swatch + 2),
            (x0 + padding + swatch, y + 2),
            color,
            thickness=-1,
        )
        cv2.putText(
            image_bgr,
            name,
            (x0 + padding + swatch + 6, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

    return image_bgr


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def apply_yolo_seg_overlay(image_bgr, da_seg_mask, ll_seg_mask):
    # Semi-opaque overlay (alpha=0.9) for mask areas.
    if da_seg_mask is not None:
        mask = da_seg_mask == 1
        image_bgr[mask] = (image_bgr[mask] * 0.1 + np.array([0, 255, 0]) * 0.9).astype(image_bgr.dtype)
    if ll_seg_mask is not None:
        mask = ll_seg_mask == 1
        image_bgr[mask] = (image_bgr[mask] * 0.1 + np.array([0, 0, 255]) * 0.9).astype(image_bgr.dtype)


def run_yolopv2(model, device, img_path, img_size, conf_thres, iou_thres, classes, agnostic_nms):
    stride = 32
    half = device.type != "cpu"
    if half:
        model.half()

    dataset = LoadImages(img_path, img_size=img_size, stride=stride)
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        _ = time_synchronized()

        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        for det in pred:
            im0 = im0s.copy()
            det_scaled = det
            if len(det_scaled):
                det_scaled[:, :4] = scale_coords(img.shape[2:], det_scaled[:, :4], im0.shape).round()
                for *xyxy, _, _ in reversed(det_scaled):
                    plot_one_box(xyxy, im0, line_thickness=3)

            apply_yolo_seg_overlay(im0, da_seg_mask, ll_seg_mask)
            return im0, det_scaled, da_seg_mask, ll_seg_mask, im0s.shape

    raise RuntimeError(f"Failed to process image {img_path} with YOLOPv2.")


def apply_yolopv2_overlay(base_image_bgr, det_scaled, da_seg_mask, ll_seg_mask, target_shape, draw_labels=False):
    output = cv2.resize(base_image_bgr, (target_shape[1], target_shape[0]))
    if det_scaled is not None and len(det_scaled):
        for *xyxy, _, _ in reversed(det_scaled):
            plot_one_box(xyxy, output, line_thickness=3)
        if draw_labels:
            for *xyxy, _, cls in reversed(det_scaled):
                cls_id = int(cls)
                if 0 <= cls_id < len(YOLO_CLASS_NAMES):
                    label = YOLO_CLASS_NAMES[cls_id]
                else:
                    label = str(cls_id)
                x1, y1 = int(xyxy[0]), int(xyxy[1])
                y_text = max(y1 - 5, 12)
                cv2.putText(
                    output,
                    label,
                    (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    label,
                    (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    apply_yolo_seg_overlay(output, da_seg_mask, ll_seg_mask)
    return output


def run_deeplabv3(model, device, image_bgr, transform_full, transform_padded, th_i_lut_trainid2color):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = transform_full(image=image_rgb)
    img_tensor = transformed["image"]

    with torch.inference_mode():
        model_input = torch.from_numpy(img_tensor).unsqueeze(0)
        model_input = model_input.to(device)
        logits = model(model_input)
        prediction = logits.argmax(axis=1)

    img_padded = transform_padded(image=image_rgb)["image"]
    prediction_color = lut.lookup_chw(
        td_u_input=prediction.byte(),
        td_i_lut=th_i_lut_trainid2color,
    ).permute((1, 2, 0)).cpu().numpy()

    # Only overlay known classes except road (train_id 1).
    pred_ids = prediction.squeeze(0).cpu().numpy()
    valid_ids = np.arange(1, 20, dtype=pred_ids.dtype)
    valid_ids = valid_ids[valid_ids != 1]
    mask = np.isin(pred_ids, valid_ids)
    
    blend = img_padded.copy()
    blend[mask] = cv2.addWeighted(
        img_padded[mask],
        0.1,
        prediction_color[mask],
        0.9,
        0.0,
    )
    blend_bgr = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
    return blend_bgr


def main():
    parser = argparse.ArgumentParser(description="Merged YOLOPv2 + DeeplabV3 demo")
    parser.add_argument("--source", type=str, default="example1.jpg", help="image file, folder, or glob")

    parser.add_argument("--yolo_weights", type=str, default="YOLOPv2/weights/yolopv2.pt", help="YOLOPv2 weights")
    parser.add_argument("--img_size", type=int, default=640, help="YOLOPv2 inference size")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="YOLOPv2 confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="YOLOPv2 IOU threshold")
    parser.add_argument("--classes", nargs="+", type=int, help="YOLOPv2 class filter")
    parser.add_argument("--agnostic_nms", action="store_true", help="YOLOPv2 class-agnostic NMS")

    default_model_dir = os.path.join("DeeplabV3", "weights", "sem_segm_gps_split")
    default_model_path = os.path.join(default_model_dir, "DeeplabV3Plus_EfficientNetB4_best_model_epoch_0060_workzone.pth")
    default_config_path = os.path.join(default_model_dir, "DeeplabV3Plus_EfficientNetB4_workzone.yaml")

    parser.add_argument("--deeplab_model_path", type=str, default=default_model_path, help="DeeplabV3 model")
    parser.add_argument("--deeplab_config_path", type=str, default=default_config_path, help="DeeplabV3 config")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--exist_ok", action="store_true", help="allow existing exp directory")

    args = parser.parse_args()
    device_arg = args.device
    if device_arg.startswith("cuda:"):
        device_arg = device_arg.split(":", 1)[1]
    device = select_device(device_arg)

    images = resolve_sources(args.source)

    output_root = os.path.join("output", "merged")
    exp_dir = increment_exp_dir(output_root, "exp1", exist_ok=args.exist_ok)
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.deeplab_config_path):
        raise FileNotFoundError(f"Config not found: {args.deeplab_config_path}")

    config = yaml.safe_load(open(args.deeplab_config_path, "r"))
    preprocess_input = get_preprocessing_fn(
        encoder_name=config["segmentation_model_backbone"],
        pretrained=config["segmentation_pretrained_dataset"],
    )

    transform_full = A.Compose([
        A.Resize(width=1280, height=720),
        A.Lambda(name="image_preprocessing", image=preprocess_input),
        A.PadIfNeeded(736, 1280),
        A.Lambda(name="to_tensor", image=to_tensor),
    ])
    transform_padded = A.Compose([
        A.Resize(width=1280, height=720),
        A.PadIfNeeded(736, 1280),
    ])

    if not os.path.exists(args.deeplab_model_path):
        raise FileNotFoundError(f"Model not found: {args.deeplab_model_path}")

    deeplab = torch.load(args.deeplab_model_path, map_location=device, weights_only=False)
    deeplab.eval()

    yolopv2 = torch.jit.load(args.yolo_weights).to(device)
    yolopv2.eval()

    th_i_lut_trainid2color = get_colormap(device)

    for image_path in images:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Error: Failed to read image: {image_path}")
            continue

        yolo_vis, det_scaled, da_seg_mask, ll_seg_mask, yolo_shape = run_yolopv2(
            yolopv2,
            device,
            image_path,
            args.img_size,
            args.conf_thres,
            args.iou_thres,
            args.classes,
            args.agnostic_nms,
        )
        deeplab_vis = run_deeplabv3(
            deeplab,
            device,
            image_bgr,
            transform_full,
            transform_padded,
            th_i_lut_trainid2color,
        )

        merged = apply_yolopv2_overlay(
            deeplab_vis,
            det_scaled,
            da_seg_mask,
            ll_seg_mask,
            yolo_shape,
        )
        merged_with_legend = apply_yolopv2_overlay(
            deeplab_vis,
            det_scaled,
            da_seg_mask,
            ll_seg_mask,
            yolo_shape,
            draw_labels=True,
        )
        merged_with_legend = draw_legend(merged_with_legend)

        stem = Path(image_path).stem
        yolo_path = exp_dir / f"{stem}_yolopv2.jpg"
        deeplab_path = exp_dir / f"{stem}_deeplabv3.jpg"
        merged_path = exp_dir / f"{stem}_merged.jpg"
        legend_path = exp_dir / f"{stem}_merged_legend.jpg"

        cv2.imwrite(str(yolo_path), yolo_vis)
        cv2.imwrite(str(deeplab_path), deeplab_vis)
        cv2.imwrite(str(merged_path), merged)
        cv2.imwrite(str(legend_path), merged_with_legend)

        print(f"Saved YOLOPv2: {yolo_path}")
        print(f"Saved DeeplabV3: {deeplab_path}")
        print(f"Saved merged: {merged_path}")
        print(f"Saved legend: {legend_path}")


if __name__ == "__main__":
    main()
