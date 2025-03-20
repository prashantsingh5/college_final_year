# src/object_removal/routes.py
import os
import json
import logging
from flask import Blueprint, request, send_file, jsonify, current_app
import uuid
from .models import HighQualityInpainter
from .utils import validate_points, ensure_directories_exist

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use an absolute path for MEDIA_FOLDER
MEDIA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'media'))
os.makedirs(MEDIA_FOLDER, exist_ok=True)

# Define the blueprint
object_removal_bp = Blueprint('object_removal', __name__, static_folder=MEDIA_FOLDER)

# Initialize the inpainter 
inpainter = None

def get_inpainter():
    """
    Lazy loading of the inpainter to avoid loading the model at import time.
    
    Returns:
        HighQualityInpainter instance
    """
    global inpainter
    if inpainter is None:
        logger.info("Initializing inpainter model...")
        inpainter = HighQualityInpainter()
    return inpainter

@object_removal_bp.route('/remove_object', methods=['POST'])
def remove_object():
    """
    API endpoint to remove objects from an image using points.
    
    Expected JSON body format:
    {
        "points": [start_x, start_y, end_x, end_y, width, height],
        "prompt": "optional custom prompt",
        "negative_prompt": "optional custom negative prompt",
        "steps": 50,
        "guidance_scale": 9.0
    }
    
    Returns:
        JSON response with result or error
    """
    try:
        # Ensure media folder exists
        if not os.path.exists(MEDIA_FOLDER):
            os.makedirs(MEDIA_FOLDER)
        
        # Get input image file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Get parameters from form or JSON
        try:
            # Try to get JSON data if present
            if request.is_json:
                data = request.json
            else:
                # Get parameters from form data
                data = {}
                for key in request.form:
                    try:
                        data[key] = json.loads(request.form[key])
                    except:
                        data[key] = request.form[key]
            
            # Extract parameters with defaults
            points = data.get('points', None)
            prompt = data.get('prompt', "completely empty space, clean blank background, perfectly clear area, matching surrounding texture and color")
            negative_prompt = data.get('negative_prompt', "any object, distortion, artifacts, noise, text, watermark")
            steps = int(data.get('steps', 50))
            guidance_scale = float(data.get('guidance_scale', 9.0))
            
            # Validate parameters
            if points is None:
                return jsonify({"error": "No points provided for object removal"}), 400
            
            validated_points = validate_points(points)
            if validated_points is None:
                return jsonify({"error": "Invalid points format"}), 400
            
        except Exception as e:
            logger.error(f"Error parsing parameters: {str(e)}")
            return jsonify({"error": f"Error parsing parameters: {str(e)}"}), 400
        
        # Generate unique filenames
        input_path = os.path.join(MEDIA_FOLDER, f'input_{uuid.uuid4().hex}.jpg')
        output_path = os.path.join(MEDIA_FOLDER, f'removed_object_{uuid.uuid4().hex}.jpg')
        
        # Save input image
        image_file.save(input_path)
        
        # Get inpainter
        inpainter = get_inpainter()
        
        # Perform inpainting
        logger.info(f"Starting object removal with points: {validated_points}")
        inpainter.inpaint(
            image_path=input_path,
            output_path=output_path,
            points=validated_points,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        )
        
        # Clean up input file
        try:
            os.remove(input_path)
        except Exception as e:
            logger.warning(f"Failed to clean up input file: {str(e)}")
        
        # Return the processed image
        return send_file(output_path, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error in remove_object endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500