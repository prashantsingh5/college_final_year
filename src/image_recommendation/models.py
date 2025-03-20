# This is the code for training the model
import os
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
import tensorflow as tf

class FeatureExtractor:
    def __init__(self, dataset_path=None, output_dir=None):
        """
        Initialize Feature Extractor
        
        Parameters:
        dataset_path (str, optional): Path to the dataset directory
        output_dir (str, optional): Directory to save model and features
        """
        # Set default paths if not provided
        if dataset_path is None:
            dataset_path = os.path.join(os.getcwd(), 'House_Room_Dataset')
        
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'image_recommendation_output')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Initialize base model
        self.base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
    def _build_model(self):
        """Build feature extraction model"""
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        model = Model(inputs=self.base_model.input, outputs=x)
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model
    
    def _build_siamese_model(self, input_shape=(224, 224, 3)):
        """Build Siamese network for embedding comparison"""
        input = Input(input_shape)
        x = self.base_model(input)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), output_shape=(128,))(x)
        model = Model(inputs=input, outputs=x)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        return model
    
    def get_embedding(self, img_path):
        """Get embedding for a single image"""
        model = self._build_model()
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return model.predict(img_array).flatten()
    
    def extract_dominant_color(self, image_path, k=10):
        """Extract dominant color from an image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return dominant_color.tolist()
    
    def calculate_contrast(self, image_path):
        """Calculate image contrast"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mean, stddev = cv2.meanStdDev(image)
        return stddev[0][0]
    
    def prepare_dataset_features(self):
        """
        Prepare features for the entire dataset
        
        Returns:
        pandas.DataFrame: DataFrame with image features
        """
        categories = ['Bedroom', 'Bathroom', 'Kitchen', 'Dinning', 'Livingroom']
        data = []
        
        for category in categories:
            category_path = os.path.join(self.dataset_path, category)
            
            if not os.path.exists(category_path):
                print(f"Warning: Category folder {category} not found. Skipping.")
                continue
            
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(category_path, img_name)
                    data.append((img_path, category))
        
        df = pd.DataFrame(data, columns=['image_path', 'category'])
        
        if df.empty:
            raise ValueError("No images found in the dataset. Please check your dataset directory.")
        
        # Calculate embeddings, dominant color, and contrast
        df['embedding'] = df['image_path'].apply(self.get_embedding)
        df['dominant_color'] = df['image_path'].apply(self.extract_dominant_color)
        df['contrast'] = df['image_path'].apply(self.calculate_contrast)
        
        return df
    
    def save_features_and_model(self):
        """
        Save dataset features and trained models
        """
        # Prepare dataset features
        df = self.prepare_dataset_features()
        
        # Paths for saving
        features_path = os.path.join(self.output_dir, 'image_features_final.csv')
        model_path = os.path.join(self.output_dir, 'siamese_model_final.h5')
        base_model_path = os.path.join(self.output_dir, 'base_model_final.h5')
        
        # Save features (with relative paths)
        df['image_path'] = df['image_path'].apply(lambda x: os.path.relpath(x))
        df.to_csv(features_path, index=False)
        
        # Save Siamese model
        siamese_model = self._build_siamese_model()
        siamese_model.save(model_path)
        
        # Save base model
        base_model = self._build_model()
        base_model.save(base_model_path)
        
        print(f"Features saved to: {features_path}")
        print(f"Siamese model saved to: {model_path}")
        print(f"Base model saved to: {base_model_path}")
        
        return df

def main():
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Save features and models
    feature_extractor.save_features_and_model()

if __name__ == '__main__':
    main()