# src/object_removal/utils.py
import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_points(points):
    """
    Validate the points format for object removal.
    
    Args:
        points: List of coordinates and dimensions
        
    Returns:
        Validated points or None if invalid
    """
    try:
        # Validate points format: [start_x, start_y, end_x, end_y, width, height]
        if not isinstance(points, list) or len(points) < 6:
            logger.error("Invalid points format. Expected [start_x, start_y, end_x, end_y, width, height]")
            return None
        
        # Ensure all values are numeric
        validated_points = [int(p) if isinstance(p, (int, float)) else 0 for p in points[:6]]
        
        # Ensure coordinates are positive
        for i, point in enumerate(validated_points):
            if point < 0:
                logger.warning(f"Negative value at index {i} corrected to 0")
                validated_points[i] = 0
        
        return validated_points
    except Exception as e:
        logger.error(f"Error validating points: {str(e)}")
        return None

def ensure_directories_exist(path):
    """
    Ensure all directories in a file path exist.
    
    Args:
        path: File path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        return False