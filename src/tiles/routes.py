from flask import Flask, jsonify, send_file, Blueprint
from PIL import Image
import os
import base64
import io

tiles_bp = Blueprint('tiles', __name__)

def get_tile_images_folder():
    # Start from the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for a folder named 'tile_images'
    for root, dirs, files in os.walk(os.path.dirname(current_dir)):
        if 'tile_images' in dirs:
            return os.path.join(root, 'tile_images')
    
    # If not found, raise an error
    raise FileNotFoundError("tile_images folder not found in project directory")

# Path to the folder containing images
IMAGE_FOLDER = get_tile_images_folder()
TILE_SIZE = (150, 150)  # Tile size for resizing (Width x Height)

@tiles_bp.route('/get-images', methods=['GET'])
def get_images():
    try:
        images_data = []
        
        # Iterate through the image folder and process images
        for idx, filename in enumerate(os.listdir(IMAGE_FOLDER)):
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
                filepath = os.path.join(IMAGE_FOLDER, filename)
                
                # Open and resize the image
                with Image.open(filepath) as img:
                    img_resized = img.resize(TILE_SIZE)
                    
                    # Convert image to base64
                    buffered = io.BytesIO()
                    img_resized.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # Append image info
                    images_data.append({
                        "index": idx + 1,
                        "image_name": filename,
                        "image_base64": f"data:image/png;base64,{img_base64}"
                    })
        
        # Limit to 10 images
        images_data = images_data[:10]
        
        # Return JSON response
        return jsonify({
            "success": True,
            "tile_size": TILE_SIZE,
            "images": images_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })
