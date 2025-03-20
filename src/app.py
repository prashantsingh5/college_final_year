from flask import Flask
from dotenv import load_dotenv
import os
from src.middleware import ip_whitelist # Import the IP whitelist middleware
from flask_cors import CORS  # Import CORS

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    
    # Load environment variables
    load_dotenv()
    
    # Configure the app
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback_secret_key')
    app.config['MEDIA_FOLDER'] = 'media'

    # Define the whitelisted IPs dynamically from the environment variables
    # WHITELISTED_IPS = set(os.getenv("WHITELISTED_IPS", "127.0.0.1").split(","))
    # Apply the middleware
    # ip_whitelist(app, WHITELISTED_IPS)

    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

    # Import and register blueprints
    from .style_transfer.routes import style_transfer_bp
    app.register_blueprint(style_transfer_bp, url_prefix='/style_transfer')
    from .final_submission.routes import final_submission_bp
    app.register_blueprint(final_submission_bp, url_prefix='/final_submission')
    from .image_recommendation.routes import image_recommendation_bp
    app.register_blueprint(image_recommendation_bp, url_prefix='/image_recommendation')
    from .wallpaper.routes import wallpaper_bp
    app.register_blueprint(wallpaper_bp, url_prefix='/wallpaper')
    from .tiles.routes import tiles_bp
    app.register_blueprint(tiles_bp, url_prefix='/tiles')

     # Register the new object removal blueprint
    from .object_removal.routes import object_removal_bp
    app.register_blueprint(object_removal_bp, url_prefix='/object_removal')

    return app
