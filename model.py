'''
Camera Classifier v0.1 Alpha - Improved Version
Copyright (c) NeuralNine

Instagram: @neuralnine
YouTube: NeuralNine
Website: www.neuralnine.com
'''

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import cv2 as cv
from PIL import Image
import PIL
import os
import pickle


class Model:
    
    def __init__(self):
        # Use SVC instead of LinearSVC for better performance with non-linear data
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        self.scaler = StandardScaler()  # Add feature scaling
        self.is_trained = False
        self.target_size = (64, 64)  # Smaller size for better performance
        self.feature_size = self.target_size[0] * self.target_size[1]
    
    def _preprocess_image(self, img):
        """Preprocess image for consistent feature extraction"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
        
        # Resize to target size
        resized_img = cv.resize(gray_img, self.target_size)
        
        # Apply histogram equalization for better contrast
        equalized_img = cv.equalizeHist(resized_img)
        
        # Apply Gaussian blur to reduce noise
        blurred_img = cv.GaussianBlur(equalized_img, (3, 3), 0)
        
        # Flatten to 1D array
        flattened_img = blurred_img.reshape(self.feature_size)
        
        return flattened_img.astype(np.float32)
    
    def train_model(self, counters):
        """Train the model with improved preprocessing and parameter tuning"""
        img_list = []
        class_list = []
        
        print("Loading and preprocessing training images...")
        
        # Load images from class 1 folder
        class1_count = 0
        for i in range(1, counters[0] + 1):
            img_path = f'1/frame{i}.jpg'
            if not os.path.exists(img_path):
                continue
                
            img = cv.imread(img_path)
            if img is None:
                continue
            
            # Preprocess the image
            processed_img = self._preprocess_image(img)
            img_list.append(processed_img)
            class_list.append(1)
            class1_count += 1
        
        # Load images from class 2 folder
        class2_count = 0
        for i in range(1, counters[1] + 1):
            img_path = f'2/frame{i}.jpg'
            if not os.path.exists(img_path):
                continue
                
            img = cv.imread(img_path)
            if img is None:
                continue
            
            # Preprocess the image
            processed_img = self._preprocess_image(img)
            img_list.append(processed_img)
            class_list.append(2)
            class2_count += 1
        
        # Safety check
        if not img_list:
            print("No images were loaded. Aborting model training.")
            return False
        
        if class1_count == 0 or class2_count == 0:
            print("Warning: One class has no images. This will cause poor performance.")
        
        print(f"Loaded {class1_count} images from class 1 and {class2_count} images from class 2")
        
        # Convert to numpy arrays
        img_list = np.array(img_list)
        class_list = np.array(class_list)
        
        # Feature scaling - very important for SVM
        print("Scaling features...")
        img_list_scaled = self.scaler.fit_transform(img_list)
        
        # Train with parameter tuning for better accuracy
        print("Training model with parameter optimization...")
        
        # If you have enough data, use grid search for optimal parameters
        if len(img_list) > 50:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly']
            }
            
            print("Performing grid search for optimal parameters...")
            grid_search = GridSearchCV(
                SVC(probability=True), 
                param_grid, 
                cv=3, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            try:
                grid_search.fit(img_list_scaled, class_list)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            except Exception as e:
                print(f"Grid search failed, using default parameters: {e}")
                self.model.fit(img_list_scaled, class_list)
        else:
            # For small datasets, use default parameters but with class balancing
            try:
                self.model = SVC(
                    kernel='rbf', 
                    C=1.0, 
                    gamma='scale', 
                    probability=True,
                    class_weight='balanced'  # Handle imbalanced classes
                )
                self.model.fit(img_list_scaled, class_list)
            except Exception as e:
                print(f"Error training model: {e}")
                return False
        
        self.is_trained = True
        print("Model successfully trained!")
        
        # Print training accuracy
        train_accuracy = self.model.score(img_list_scaled, class_list)
        print(f"Training accuracy: {train_accuracy:.3f}")
        
        return True
    
    def predict(self, frame):
        """Make prediction with confidence score"""
        if not self.is_trained:
            print("Model not trained yet!")
            return None
        
        # Handle different frame formats
        if isinstance(frame, tuple):
            if len(frame) >= 2 and frame[1] is not None:
                frame = frame[1]
            elif len(frame) >= 1 and frame[0] is not None:
                frame = frame[0]
            else:
                print("Invalid frame tuple provided")
                return None
        
        if frame is None:
            print("Invalid frame provided for prediction")
            return None
        
        try:
            # Preprocess the frame
            processed_frame = self._preprocess_image(frame)
            
            # Scale the features
            scaled_frame = self.scaler.transform([processed_frame])
            
            # Make prediction
            prediction = self.model.predict(scaled_frame)[0]
            
            # Get prediction probabilities for confidence
            probabilities = self.model.predict_proba(scaled_frame)[0]
            confidence = np.max(probabilities)
            
            print(f"Prediction: Class {prediction}, Confidence: {confidence:.3f}")
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_prediction_confidence(self, frame):
        """Get prediction with confidence score"""
        if not self.is_trained:
            return None, 0.0
        
        # Handle different frame formats
        if isinstance(frame, tuple):
            if len(frame) >= 2 and frame[1] is not None:
                frame = frame[1]
            elif len(frame) >= 1 and frame[0] is not None:
                frame = frame[0]
            else:
                return None, 0.0
        
        if frame is None:
            return None, 0.0
        
        try:
            # Preprocess the frame
            processed_frame = self._preprocess_image(frame)
            
            # Scale the features
            scaled_frame = self.scaler.transform([processed_frame])
            
            # Make prediction
            prediction = self.model.predict(scaled_frame)[0]
            probabilities = self.model.predict_proba(scaled_frame)[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, 0.0
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        if not self.is_trained:
            print("No trained model to save")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'target_size': self.target_size,
                'feature_size': self.feature_size
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.target_size = model_data['target_size']
            self.feature_size = model_data['feature_size']
            self.is_trained = True
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False