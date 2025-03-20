import os

# Use an absolute path for MEDIA_FOLDER
MEDIA_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'media'))

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_user_folder_structure(user_id):
    """Create user folder structure if it doesn't exist"""
    user_folder = os.path.join(MEDIA_FOLDER, user_id)
    input_folder = os.path.join(user_folder, 'input')
    output_folder = os.path.join(user_folder, 'output')

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    return input_folder, output_folder

def get_last_saved_image(output_folder):
    """Retrieve the most recently saved image from the output folder"""
    files = [f for f in os.listdir(output_folder) if allowed_file(f)]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(output_folder, x)), reverse=True)
    return os.path.join(output_folder, files[0])

def get_next_output_number(output_folder):
    """Get the next available output number"""
    # Get all files that start with 'output' and have a valid image extension
    existing_files = [f for f in os.listdir(output_folder) 
                     if f.startswith('output') and 
                     any(f.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])]
    
    if not existing_files:
        return 1
    
    # Extract numbers from filenames
    numbers = []
    for filename in existing_files:
        # Remove 'output' prefix and file extension
        name_part = os.path.splitext(filename)[0]
        if name_part[6:].isdigit():  # Check if remaining part is a number
            numbers.append(int(name_part[6:]))
    
    return max(numbers) + 1 if numbers else 1