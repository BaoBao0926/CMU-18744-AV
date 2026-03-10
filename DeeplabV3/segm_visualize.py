import torch
import torchvision
import numpy as np
import cv2
import misc.segm.lookup_table as lut
from collections import namedtuple
import os
import albumentations as A
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import yaml
import argparse
import sys


class DatasetWorkzoneSemantic(torchvision.datasets.Cityscapes):

    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),

        CityscapesClass(  'Road'                 ,  1 ,        1 , 'flat'            , 1       , False        , False        , ( 70, 70, 70) ),
        CityscapesClass(  'Sidewalk'             ,  2 ,        2 , 'flat'            , 1       , False        , False        , (102,102,156) ),
        CityscapesClass(  'Bike Lane'            ,  3 ,        3 , 'flat'            , 1       , False        , False        , (190,153,153) ),
        CityscapesClass(  'Off-Road'             ,  4 ,        4 , 'flat'            , 1       , False        , False        , (180,165,180) ),
        CityscapesClass(  'Roadside'             ,  5 ,        5 , 'flat'            , 1       , False        , False        , (150,100,100) ),

        CityscapesClass(  'Barrier'              ,  6 ,        6 , 'barrier'         , 2       , False        , False        , (246, 116, 185) ),
        CityscapesClass(  'Barricade'            ,  7 ,        7 , 'barrier'         , 2       , False        , False        , (248, 135, 182) ),
        CityscapesClass(  'Fence'                ,  8 ,        8 , 'barrier'         , 2       , False        , False        , (251, 172, 187) ),

        CityscapesClass(  'Police Vehicle'       ,  9 ,        9 , 'vehicle'         , 3       , True         , False        , (255, 68, 51) ),
        CityscapesClass(  'Work Vehicle'         ,  10,        10, 'vehicle'         , 3       , True         , False        , (255,104, 66) ),

        CityscapesClass(  'Police Officer'       ,  11,        11, 'human'           , 4       , True         , False        , (184, 107, 35) ),
        CityscapesClass(  'Worker'               ,  12,        12, 'human'           , 4       , True         , False        , (205, 135, 29) ),

        CityscapesClass(  'Cone'                 ,  13,        13, 'object'          , 5       , True         , False        , (30, 119, 179) ),
        CityscapesClass(  'Drum'                 ,  14,        14, 'object'          , 5       , True         , False        , (44, 79, 206) ),
        CityscapesClass(  'Vertical Panel'       ,  15,        15, 'object'          , 5       , True         , False        , (102, 81, 210) ),
        CityscapesClass(  'Tubular Marker'       ,  16,        16, 'object'          , 5       , True         , False        , (170, 118, 213) ),
        CityscapesClass(  'Work Equipment'       ,  17,        17, 'object'          , 5       , True         , False        , (214, 154, 219) ),

        CityscapesClass(  'Arrow Board'          ,  18,        18, 'guidance'        , 6       , True         , False        , (241, 71, 14) ),
        CityscapesClass(  'TTC Sign'             ,  19,        19, 'guidance'        , 6       , True         , False        , (254, 139, 32) ),
    ]

    def __init__(
            self,
            root,
            device,
            split: str = "train",
            mode: str = "fine",
            target_type = "semantic",
            transform = None,
            target_transform = None,
            transforms = None,
            small_size = False
        ) -> None:
        ## don't want to call Cityscapes.__init__
        ## instead want to call VisionDataset.__init__
        super(DatasetWorkzoneSemantic.__bases__[0], self).__init__(root, transforms, transform, target_transform)
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "images", split)
        if mode == "fine":
            self.targets_dir = os.path.join(self.root, "gtFine", split)
        else:
            self.targets_dir = os.path.join(self.root, "gtCoarse", split)
        self.target_type = [ target_type ]
        self.split = split
        self.images = []
        self.targets = []

        # verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "val")
        else:
            valid_modes = ("train", "val")


        for file_name in os.listdir(self.images_dir):
            target_types = []
            for t in self.target_type:
                target_name = "{}{}".format(
                    os.path.splitext(file_name)[0], self._get_target_suffix(self.mode, t)
                )
                target_types.append(os.path.join(self.targets_dir, target_name))

            self.images.append(os.path.join(self.images_dir, file_name))
            self.targets.append(target_types)

        self.device = device
        # setup lookup tables for class/color conversions
        l_key_id, l_key_trainid, l_key_color = self._get_class_properties()
        ar_u_key_id = np.asarray(l_key_id, dtype = np.uint8)
        ar_u_key_trainid = np.asarray(l_key_trainid, dtype = np.uint8)
        ar_u_key_color = np.asarray(l_key_color, dtype = np.uint8)
        _, self.th_i_lut_id2trainid = lut.get_lookup_table(
            ar_u_key = ar_u_key_id,
            ar_u_val = ar_u_key_trainid,
            v_val_default = 0,  # default class is 0 - unlabeled
            device = self.device,
        )
        _, self.th_i_lut_trainid2id = lut.get_lookup_table(
            ar_u_key = ar_u_key_trainid,
            ar_u_val = ar_u_key_id,
            v_val_default = 0,  # default class is 0 - unlabeled
            device = self.device,
        )
        _, self.th_i_lut_trainid2color = lut.get_lookup_table(
            ar_u_key = ar_u_key_trainid,
            ar_u_val = ar_u_key_color,
            v_val_default = 0,  # default color is black
            device = self.device,
        )
    
    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"_instanceIds.png"
        elif target_type == "semantic":
            return f"_labelIds.png"
        elif target_type == "color":
            return f"_color.png"
        else:
            raise ValueError(f"Unknown value '{target_type}' for argument target_type.")

    def _get_class_properties(self):
        # iterate over named tuples (nt)
        l_key_id = list()
        l_key_trainid = list()
        l_key_color = list()
        # append classes
        for nt_class in self.classes:
            if nt_class.train_id in [-1, 255]:
                continue
            l_key_id.append([nt_class.id])
            l_key_trainid.append([nt_class.train_id])
            l_key_color.append(nt_class.color)
        # append class background
        l_key_id.append([0])
        l_key_trainid.append([0])
        l_key_color.append([0, 0, 0])
        return l_key_id, l_key_trainid, l_key_color

    def __getitem__(self, index):
        # read images
        p_image = self.images[index]
        p_target = self.targets[index][0]  # 0 is index of semantic target type
        image = cv2.imread(p_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = cv2.imread(p_target, cv2.IMREAD_UNCHANGED)
        if self.transform is not None:
            transformed = self.transform(image = image, mask = target)
            image = transformed["image"]
            target = transformed["mask"]
        return image, target, p_image, p_target

def get_colormap(device):
    dataset_dummy = DatasetWorkzoneSemantic
    l_key_id = list()
    l_key_trainid = list()
    l_key_color = list()
    # append classes
    for nt_class in dataset_dummy.classes:
        if nt_class.train_id in [-1, 255]:
            continue
        l_key_id.append([nt_class.id])
        l_key_trainid.append([nt_class.train_id])
        l_key_color.append(nt_class.color)
    # append class background
    l_key_id.append([0])
    l_key_trainid.append([0])
    l_key_color.append([0, 0, 0])
    
    ar_u_key_trainid = np.asarray(l_key_trainid, dtype = np.uint8)
    ar_u_key_color = np.asarray(l_key_color, dtype = np.uint8)
    
    _, th_i_lut_trainid2color = lut.get_lookup_table(
        ar_u_key = ar_u_key_trainid,
        ar_u_val = ar_u_key_color,
        v_val_default = 0,  # default color is black
        device = device,
    )
    return th_i_lut_trainid2color

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")

def main():
    parser = argparse.ArgumentParser(description="Roadwork Segmentation Visualization")
    parser.add_argument("--image_path", type=str, default="example5.jpg", help="Path to the single input image")
    
    # Defaults constructed using os.path.join to avoid escape character issues (like \r)
    default_model_dir = os.path.join("DeeplabV3", "weights", "sem_segm_gps_split")
    default_model_path = os.path.join(default_model_dir, "DeeplabV3Plus_EfficientNetB4_best_model_epoch_0060_workzone.pth")
    default_config_path = os.path.join(default_model_dir, "DeeplabV3Plus_EfficientNetB4_workzone.yaml")

    parser.add_argument("--model_path", type=str, default=default_model_path, 
                        help="Path to the model .pth file. If not provided, will look in default locations.")
    parser.add_argument("--config_path", type=str, default=default_config_path,
                        help="Path to the model config .yaml file. If not provided, will look in default locations.")
                        
    parser.add_argument("--output_path", type=str, default="example_segm_vis.jpg",
                        help="Path to save output visualization. If not provided, will be derived from input filename in './output/'")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (e.g. 'cpu', 'cuda:0')")

    args = parser.parse_args()
    device = args.device
    
    print(f"Device: {device}")
    
    # Model Setup
    model_path = args.model_path
    config_path = args.config_path

    # Auto-detection logic if paths are not provided
    if model_path is None or config_path is None:
        model_base_path = "./models/sem_segm_gps_split/" 
        if not os.path.exists(model_base_path):
            sibling_path = os.path.join("..", "ROADwork", "models", "sem_segm_gps_split")
            if os.path.exists(sibling_path):
                model_base_path = sibling_path
        
        if model_path is None:
            model_path = os.path.join(model_base_path, "DeeplabV3Plus_EfficientNetB4_best_model_epoch_0060_workzone.pth")
        if config_path is None:
            config_path = os.path.join(model_base_path, "DeeplabV3Plus_EfficientNetB4_workzone.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Loading config from {config_path}")
    config = yaml.safe_load(open(config_path, "r"))

    preprocess_input = get_preprocessing_fn(
        encoder_name = config["segmentation_model_backbone"],
        pretrained = config["segmentation_pretrained_dataset"],
    )

    transform_full = A.Compose([
        A.Resize(width=1280, height=720),
        A.Lambda(name = "image_preprocessing", image = preprocess_input),
        A.PadIfNeeded(736, 1280),
        A.Lambda(name = "to_tensor", image = to_tensor),
    ])

    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
         print(f"Error: Model file not found at {model_path}")
         return

    # Set weights_only=False to allow loading full model object (needed for older pytorch/smp versions or full model dumps)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Process single image
    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image using cv2: {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    # Since we are not using the dataset class, we call transform directly
    # transform expects 'image' and optional 'mask'. We only have image.
    transformed = transform_full(image=image)
    img_tensor = transformed["image"]

    # Inference
    with torch.inference_mode():
        model_input = torch.from_numpy(img_tensor).unsqueeze(0)
        model_input = model_input.to(device)
        logits = model(model_input)
        prediction = logits.argmax(axis = 1)

    # Visualization
    # Setup lookup table
    th_i_lut_trainid2color = get_colormap(device)

    # Load original image again for blending (or reuse if we want resized version)
    # The original logic used `img_padded` which was resized and padded.
    # We should reconstruct that part for consistent visualization with the model input dimension.
    transform_padded = A.Compose([
        A.Resize(width=1280, height=720),
        A.PadIfNeeded(736, 1280),
    ])
    # Need to reload or just transform the original numpy array before preprocessing
    # But wait, 'image' variable already has RGB loaded.
    img_padded = transform_padded(image = image)["image"] # This returns numpy array (H, W, C)

    prediction_color = lut.lookup_chw(
        td_u_input = prediction.byte(),
        td_i_lut = th_i_lut_trainid2color,
    ).permute((1, 2, 0)).cpu().numpy()
    
    # Blending
    blend = cv2.addWeighted(img_padded, 0.4, prediction_color, 0.6, 0.0)
    
    # Convert RGB back to BGR for saving with cv2
    blend_bgr = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)

    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_dir = "./output/segm_gps_split/vis"
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, img_name)
    
    # Ensure directory exists if specific path provided
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    cv2.imwrite(output_path, blend_bgr)
    print(f"Saved prediction visualization to {output_path}")

if __name__ == "__main__":
    main()

