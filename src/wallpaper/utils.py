import os
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
from segment_anything import build_sam, SamPredictor
from .GroundingDINO.groundingdino.util import box_ops
from .GroundingDINO.groundingdino.util.slconfig import SLConfig
from .GroundingDINO.groundingdino.util.utils import clean_state_dict
from .GroundingDINO.groundingdino.util.inference import load_image, predict
from .GroundingDINO.groundingdino.models import build_model
from huggingface_hub import hf_hub_download

MEDIA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'media'))

def get_sam_weights_path():
    # Start from the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search for sam_vit_h_4b8939.pth in parent directories and subdirectories
    for root, dirs, files in os.walk(os.path.dirname(current_dir)):
        if 'sam_vit_h_4b8939.pth' in files:
            return os.path.join(root, 'sam_vit_h_4b8939.pth')
    
    # If not found, raise an error
    raise FileNotFoundError("SAM weights file 'sam_vit_h_4b8939.pth' not found in project directory")

# Replace the hardcoded path wherever SAM checkpoint is used with this function call
SAM_CHECKPOINT = get_sam_weights_path()

# Function to load the GroundingDINO model
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cuda'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.to(device)
    model.eval()
    return model

# Generate wall mask function
def generate_wall_mask(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load GroundingDINO model
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device=device)

    # Load SAM model
    sam_model = build_sam(checkpoint=SAM_CHECKPOINT)
    sam_model.to(device)
    sam_predictor = SamPredictor(sam_model)

    # Load image
    image_source, image = load_image(image_path)
    sam_predictor.set_image(image_source)

    # Detect walls using GroundingDINO
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=image,
        caption="wall",
        box_threshold=0.35,
        text_threshold=0.25
    )

    if len(boxes) == 0:
        return None

    # Generate mask using SAM
    boxes = boxes.to(device)
    H, W, _ = image_source.shape
    scaling_tensor = torch.tensor([W, H, W, H], device=device)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * scaling_tensor
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )

    wall_mask = masks[0][0].cpu().numpy()
    binary_mask = (wall_mask > 0).astype(np.uint8) * 255
    return binary_mask

# Overlay wallpaper function
def overlay_wallpaper(room_image_path, wallpaper_image_path, mask):
    room_image = cv2.imread(room_image_path)
    wallpaper = cv2.imread(wallpaper_image_path)

    if mask.shape[:2] != room_image.shape[:2]:
        mask = cv2.resize(mask, (room_image.shape[1], room_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_3channel = cv2.merge([mask, mask, mask])
    wall_indices = np.where(mask > 0)
    min_row, max_row = min(wall_indices[0]), max(wall_indices[0])
    min_col, max_col = min(wall_indices[1]), max(wall_indices[1])
    wall_height = max_row - min_row
    wall_width = max_col - min_col

    tiled_wallpaper = np.tile(
        wallpaper,
        (
            wall_height // wallpaper.shape[0] + 1,
            wall_width // wallpaper.shape[1] + 1,
            1
        )
    )
    tiled_wallpaper_cropped = tiled_wallpaper[:wall_height, :wall_width]
    wallpaper_overlay = np.zeros_like(room_image)
    wallpaper_overlay[min_row:max_row, min_col:max_col] = tiled_wallpaper_cropped

    result = np.where(mask_3channel == 255, wallpaper_overlay, room_image)
    return result
