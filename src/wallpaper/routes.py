import os
import cv2
import numpy as np
import torch
import logging
from flask import Blueprint, request, send_file, jsonify, redirect, current_app
from segment_anything import build_sam, SamPredictor
from .GroundingDINO.groundingdino.util import box_ops
from .GroundingDINO.groundingdino.util.slconfig import SLConfig
from .GroundingDINO.groundingdino.util.utils import clean_state_dict
from .GroundingDINO.groundingdino.util.inference import load_image, predict
from .GroundingDINO.groundingdino.models import build_model
from huggingface_hub import hf_hub_download
from .utils import load_model_hf, generate_wall_mask, overlay_wallpaper

logging.basicConfig(level=logging.DEBUG)

# Use an absolute path for MEDIA_FOLDER
MEDIA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'media'))

wallpaper_bp = Blueprint('wallpaper', __name__, static_folder=MEDIA_FOLDER)

# Ensure the media folder exists
os.makedirs(MEDIA_FOLDER, exist_ok=True)

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

@wallpaper_bp.route('/apply_wallpaper', methods=['POST'])
def apply_wallpaper():
    try:
        # Ensure media folder exists
        if not os.path.exists(MEDIA_FOLDER):
            os.makedirs(MEDIA_FOLDER)

        # Get input files
        room_image_file = request.files.get('room_image')
        wallpaper_image_file = request.files.get('wallpaper_image')

        if room_image_file is None or wallpaper_image_file is None:
            return jsonify({"error": "Missing room_image or wallpaper_image in the request."}), 400

        # Save uploaded files temporarily
        room_image_path = os.path.join(MEDIA_FOLDER, 'temp_room_image.jpg')
        wallpaper_image_path = os.path.join(MEDIA_FOLDER, 'temp_wallpaper_image.jpg')
        room_image_file.save(room_image_path)
        wallpaper_image_file.save(wallpaper_image_path)

        # Verify if files were saved
        if not os.path.exists(room_image_path) or not os.path.exists(wallpaper_image_path):
            return jsonify({"error": "Failed to save uploaded files."}), 500

        # Generate wall mask
        wall_mask = generate_wall_mask(room_image_path)
        if wall_mask is None:
            return jsonify({"error": "No walls detected in the image."}), 400

        # Apply wallpaper
        final_image = overlay_wallpaper(room_image_path, wallpaper_image_path, wall_mask)

        # Save the final output
        output_path = os.path.join(MEDIA_FOLDER, 'room_with_wallpaper.jpg')
        cv2.imwrite(output_path, final_image)

        # Remove temporary files
        os.remove(room_image_path)
        os.remove(wallpaper_image_path)

        # Return the final image in the response
        return send_file(output_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
