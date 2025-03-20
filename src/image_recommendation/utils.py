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
from dotenv import load_dotenv

# Get the absolute path of the project root
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

print(f"Utils - Project Root: {PROJECT_ROOT}")
print(f"Utils - Image Base Dir: {IMAGE_BASE_DIR}")
print(f"Utils - CSV Path: {CSV_PATH}")
print(f"Utils - CSV exists: {os.path.exists(CSV_PATH)}")

df = pd.read_csv(CSV_PATH)

# Function to preprocess and get embeddings
def get_embedding(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model.predict(img_array).flatten()

# Function to extract the dominant color of an image
def extract_dominant_color(image_array, k=10):
    image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color.tolist()

# Function to calculate the contrast of an image
def calculate_contrast(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    contrast = stddev[0][0]
    return contrast

# Function to compute color similarity using CIEDE2000
def rgb_to_lab(rgb):
    r, g, b = rgb[0] / 255, rgb[1] / 255, rgb[2] / 255
    r = r if r > 0.04045 else r / 12
    g = g if g > 0.04045 else g / 12
    b = b if b > 0.04045 else b / 12
    r = r ** (1 / 2.4) if r > 0.0031308 else 7.787 * r + 16 / 116
    g = g ** (1 / 2.4) if g > 0.0031308 else 7.787 * g + 16 / 116
    b = b ** (1 / 2.4) if b > 0.0031308 else 7.787 * b + 16 / 116
    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    l = 116 * y - 16
    a = 500 * (x - y)
    b = 200 * (y - z)
    return l, a, b

def delta_e_cie2000(lab1, lab2):
    dl = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    c1 = (lab1[1] ** 2 + lab1[2] ** 2) ** 0.5
    c2 = (lab2[1] ** 2 + lab2[2] ** 2) ** 0.5
    sl = 1
    sc = 1 + 0.045 * c1
    sh = 1 + 0.015 * c1
    delta_l = dl / sl
    delta_c = (da ** 2 + db ** 2) ** 0.5 / sc
    delta_h = delta_c ** 2 + delta_l ** 2 - delta_l * delta_c * (lab1[1] * lab2[1] + lab1[2] * lab2[2]) / (c1 * c2)
    return delta_h ** 0.5

def color_similarity(color1, color2):
    if isinstance(color1, str) and color1[0] == '[' and color1[-1] == ']':
        color1 = color1[1:-1].replace(' ', '').split(',')
    if isinstance(color2, str) and color2[0] == '[' and color2[-1] == ']':
        color2 = color2[1:-1].replace(' ', '').split(',')
    color1 = [float(x) for x in color1]
    color2 = [float(x) for x in color2]
    lab1 = rgb_to_lab(color1)
    lab2 = rgb_to_lab(color2)
    delta_e = delta_e_cie2000(lab1, lab2)
    return 1 / (1 + delta_e)

# Function to recommend images based on ResNet features, dominant color, and contrast
def recommend_images(uploaded_embedding, uploaded_dominant_color, uploaded_contrast, category, top_n=5):
    # Filter the dataset for the given category
    category_df = df[df['category'] == category]

    # Calculate similarities
    category_embeddings = category_df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), dtype=float, sep=' '))
    feature_similarities = category_embeddings.apply(lambda x: cosine_similarity([uploaded_embedding], [x]).flatten()[0])

    category_dominant_colors = np.array([eval(color) for color in category_df['dominant_color']])
    color_similarities = np.array([color_similarity(uploaded_dominant_color, color) for color in category_dominant_colors])

    category_contrasts = np.array(category_df['contrast'].tolist())
    contrast_similarities = 1 / (1 + np.abs(category_contrasts - uploaded_contrast))

    # Combine all similarities
    combined_similarities = (feature_similarities * 0.4 + color_similarities * 0.3 + contrast_similarities * 0.3)

    # Get top N recommendations
    top_indices = combined_similarities.argsort()[-top_n:][::-1]
    return category_df.iloc[top_indices]['image_path'].tolist()

# Function to convert an image to base64 string
def normalize_path(path):
    # Convert Windows path separators to Unix-style
    normalized = path.replace('\\', '/')
    # Join with base directory using os.path.join
    return os.path.join(IMAGE_BASE_DIR, normalized)

# Rest of your imports and model setup remains the same...

def image_to_base64(image_path):
    try:
        # Normalize the path
        full_image_path = normalize_path(image_path)
        
        # Print path for debugging
        print(f"Attempting to open: {full_image_path}")
        
        # Ensure the file exists
        if not os.path.exists(full_image_path):
            print(f"File not found: {full_image_path}")
            raise FileNotFoundError(f"Image not found: {full_image_path}")
            
        # Read and encode the image
        with open(full_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        raise
