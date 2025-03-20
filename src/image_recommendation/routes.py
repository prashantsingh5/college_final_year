import os
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify, Blueprint
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import base64
from io import BytesIO
from PIL import Image
from .utils import get_embedding, extract_dominant_color, calculate_contrast, rgb_to_lab, delta_e_cie2000, color_similarity, recommend_images, image_to_base64
from dotenv import load_dotenv

image_recommendation_bp = Blueprint('image_recommendation', __name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Construct the path to the image_recommendation folder
IMAGE_BASE_DIR = os.path.join(PROJECT_ROOT, 'src', 'image_recommendation')

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense layer on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the precomputed features CSV dynamically
CSV_PATH = os.path.join(IMAGE_BASE_DIR, "image_features_final.csv")
# Debug prints
print(f"Project Root: {PROJECT_ROOT}")
print(f"Image Base Dir: {IMAGE_BASE_DIR}")
print(f"CSV Path: {CSV_PATH}")
print(f"CSV exists: {os.path.exists(CSV_PATH)}")

df = pd.read_csv(CSV_PATH)

@image_recommendation_bp.route('/recommend', methods=['POST'])
def recommend():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    category = request.form.get('category', 'Bedroom')  # Default to 'Bedroom' if not provided

    # Read and preprocess the uploaded image
    img = Image.open(file.stream)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Get features for the uploaded image
    uploaded_embedding = get_embedding(img_array)
    uploaded_dominant_color = extract_dominant_color(img_array)
    uploaded_contrast = calculate_contrast(img_array)

    # Get recommendations
    recommended_image_paths = recommend_images(uploaded_embedding, uploaded_dominant_color, uploaded_contrast, category)

    # Convert recommended images to base64
    recommended_images = [
        {
            'image': f'data:image/jpeg;base64,{image_to_base64(path)}',  # Complete base64 encoded string
            'path': path
        }
        for path in recommended_image_paths
    ]

    # Ensure the response sends the full base64 strings
    return jsonify({
        'recommended_images': recommended_images,
        'dominant_color': uploaded_dominant_color,
        'contrast': float(uploaded_contrast)
    })
