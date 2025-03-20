import os
import logging
from flask import Blueprint, request, send_file, jsonify, redirect, current_app
from werkzeug.utils import secure_filename
from .utils import allowed_file, create_user_folder_structure, get_last_saved_image, get_next_output_number
from .image_processor_final import process_image, detect_objects, process_image_for_inpainting

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Use an absolute path for MEDIA_FOLDER
MEDIA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'media'))

final_submission_bp = Blueprint('final_submission', __name__, static_folder=MEDIA_FOLDER)

# Ensure the media folder exists
os.makedirs(MEDIA_FOLDER, exist_ok=True)

@final_submission_bp.route('/wall_color_change', methods=['POST'])
def wall_color_change():
    # Get user_id from form-data
    user_id = request.form.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user using the absolute MEDIA_FOLDER path
    input_folder, output_folder = create_user_folder_structure(user_id)
    
    try:
        if 'image' not in request.files:
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            return redirect(request.url)
        
        text_prompt = request.form.get('text_prompt')
        color_name = request.form.get('color_name')
        
        if not text_prompt or not color_name:
            return "Error: Missing text prompt or color name", 400
        
        filename = secure_filename(image.filename)
        image_path = os.path.join(input_folder, filename)
        image.save(image_path)
        
        # Extract original file extension
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Preserve the original extension if it's a supported format
        output_ext = file_ext if file_ext in ['.jpg', '.jpeg', '.png'] else '.jpg'
        
        next_number = get_next_output_number(output_folder)
        output_filename = f'output{next_number}{output_ext}'
        output_path = os.path.join(output_folder, output_filename)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        process_image(image_path, text_prompt, color_name=color_name, output_path=output_path)
        
        # Determine mimetype based on file extension
        mimetype = 'image/jpeg' if output_ext in ['.jpg', '.jpeg'] else 'image/png'
        
        return send_file(output_path, mimetype=mimetype, as_attachment=True, download_name=output_filename)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}", 500

@final_submission_bp.route('/detect_objects', methods=['POST'])
def detect_objects_route():
    # Get user_id from form-data
    user_id = request.form.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user
    input_folder, output_folder = create_user_folder_structure(user_id)

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(input_folder, filename)
        file.save(file_path)

        detected_objects = detect_objects(file_path, return_objects=True)
        return jsonify({"detected_objects": detected_objects})
    else:
        return jsonify({"error": "Invalid file type"}), 400

@final_submission_bp.route('/inpaint', methods=['POST'])
def inpaint_image():
    # Get user_id from form-data
    user_id = request.form.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user
    input_folder, output_folder = create_user_folder_structure(user_id)

    if 'image' not in request.files:
        return 'No image file uploaded', 400
    
    image_file = request.files['image']
    object_to_detect = request.form.get('object_to_detect')
    inpaint_prompt = request.form.get('inpaint_prompt')
    
    if not object_to_detect or not inpaint_prompt:
        return 'Missing object_to_detect or inpaint_prompt', 400

    image_path = os.path.join(input_folder, secure_filename(image_file.filename))
    image_file.save(image_path)

    inpainted_image = process_image_for_inpainting(image_path, object_to_detect, inpaint_prompt)

    next_number = get_next_output_number(output_folder)
    output_filename = f"output{next_number}.png"
    output_path = os.path.join(output_folder, output_filename)
    inpainted_image.save(output_path)

    return send_file(output_path, mimetype='image/png', as_attachment=True, download_name=output_filename)

@final_submission_bp.route('/inpaint_using_last_output', methods=['POST'])
def inpaint_using_last_output():
    # Try to retrieve user_id from form data or JSON data
    user_id = request.form.get('user_id') or request.json.get('user_id')
    if not user_id:
        return "Error: Missing user_id", 400

    # Create folders for the user
    _, output_folder = create_user_folder_structure(user_id)

    # Get the last saved image in the output folder
    last_image_path = get_last_saved_image(output_folder)
    if not last_image_path:
        return "Error: No previously processed image found for this user", 404

    # Retrieve the inpainting parameters from either form or JSON data
    object_to_detect = request.form.get('object_to_detect') or request.json.get('object_to_detect')
    inpaint_prompt = request.form.get('inpaint_prompt') or request.json.get('inpaint_prompt')
    
    if not object_to_detect or not inpaint_prompt:
        return 'Missing object_to_detect or inpaint_prompt', 400

    # Process the image for inpainting
    inpainted_image = process_image_for_inpainting(last_image_path, object_to_detect, inpaint_prompt)

    # Get the next output number and create the new filename
    next_number = get_next_output_number(output_folder)
    output_filename = f"output{next_number}.png"
    output_path = os.path.join(output_folder, output_filename)
    inpainted_image.save(output_path)

    return send_file(output_path, mimetype='image/png', as_attachment=True, download_name=output_filename)
