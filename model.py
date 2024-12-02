# model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import string
import asyncio
import random
from ultralytics import YOLO
import cv2
import base64
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

class BeauSkinModel:
    def __init__(self, models_dir: str):
        try:
            self.models_dir = Path(models_dir)
            
            # Load models
            print("Loading models...")
            self.acne_model = tf.keras.models.load_model(self.models_dir / 'acne_grade_model.h5')
            self.skin_type_model = tf.keras.models.load_model(self.models_dir / 'skintypes_detection_model.h5')
            self.acne_types_model = YOLO(str(self.models_dir / 'best_model.pt'))
            self.chatbot_model = tf.keras.models.load_model(self.models_dir / 'chatbot_model.h5')
            
            # Load and process chatbot data
            print("Loading chatbot files...")
            with open(self.models_dir / 'skin_treatment.json') as file:
                self.data = json.load(file)
            
            # Prepare responses dictionary
            self.responses = {}
            inputs = []
            tags = []
            for intent in self.data['intents']:
                self.responses[intent['tag']] = intent['responses']
                for lines in intent['input']:
                    inputs.append(lines)
                    tags.append(intent['tag'])
            
            # Initialize tokenizer and fit on inputs
            self.tokenizer = Tokenizer(num_words=2000)
            # Clean inputs
            cleaned_inputs = [self.clean_text(text) for text in inputs]
            self.tokenizer.fit_on_texts(cleaned_inputs)
            
            # Initialize and fit label encoder
            self.le = LabelEncoder()
            self.le.fit(tags)
            
            # Get input shape from training data
            train = self.tokenizer.texts_to_sequences(cleaned_inputs)
            x_train = pad_sequences(train)
            self.input_shape = x_train.shape[1]
            
            # Define fixed colors for YOLO labels
            self.label_colors = {
                'blackheads': (255, 0, 0),    # Red
                'dark spot': (0, 255, 0),     # Green
                'nodules': (0, 0, 255),       # Blue
                'papules': (255, 255, 0),     # Yellow
                'pustules': (0, 255, 255),    # Cyan
                'whiteheads': (255, 0, 255)   # Magenta
            }
            
            print("All models and data loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def clean_text(self, text):
        """Clean text by removing punctuation and converting to lowercase"""
        text = ''.join([letter.lower() for letter in text if letter not in string.punctuation])
        return text
    
    async def get_chatbot_response(self, text):
        """Get chatbot response using the new model"""
        try:
            # Clean and preprocess input text
            cleaned_text = self.clean_text(text)
            texts_p = [cleaned_text]
            
            # Tokenize and pad input
            prediction_input = self.tokenizer.texts_to_sequences(texts_p)
            prediction_input = np.array(prediction_input).reshape(-1)
            prediction_input = pad_sequences([prediction_input], self.input_shape)
            
            # Get prediction
            output = self.chatbot_model.predict(prediction_input)
            output = output.argmax()
            
            # Get response tag and random response
            response_tag = self.le.inverse_transform([output])[0]
            response = random.choice(self.responses[response_tag])
            
            return {
                "status": "success",
                "response": str(response),
                "intent": response_tag
            }
        except Exception as e:
            print(f"Chatbot error: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def preprocess_image(self, image, target_size=(128, 128)):
        """Preprocess image for model predictions"""
        if isinstance(image, str):  # If path is provided
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def get_acne_label_name(self, prediction):
        """Get acne condition label name"""
        class_labels = {0: "Mild", 1: "Moderate", 2: "No acne", 3: "Severe"}
        predicted_index = prediction.argmax()
        return class_labels.get(predicted_index, "Unknown")

    def get_skin_type_label_name(self, label):
        """Get skin type label name"""
        skin_type_labels = {0: 'dry', 1: 'normal', 2: 'oily'}
        return skin_type_labels.get(label, "Unknown")

    def process_yolo_results(self, results, img_rgb):
        """Process YOLO detection results and draw on image"""
        detected_acne_types = []
        
        # Process each bounding box
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            label = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            label_name = self.acne_types_model.names[label]
            color = self.label_colors.get(label_name, (255, 255, 255))
            
            if label_name not in detected_acne_types:
                detected_acne_types.append(label_name)
            
            # Draw bounding box and label
            cv2.rectangle(img_rgb, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        color, 2)
            cv2.putText(img_rgb, 
                       f'{label_name}', 
                       (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        return img_rgb, detected_acne_types

    async def analyze_image(self, image_path):
        try:
            # Read image from path
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for OpenCV processing
            img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Run YOLO detection
            results = self.acne_types_model.predict(img_rgb)
            
            # Process YOLO results and draw on image
            annotated_image, detected_acne_types = self.process_yolo_results(results, img_rgb)
            
            # Preprocess image for other models
            preprocessed_image = self.preprocess_image(image)
            
            # Get acne grade prediction
            acne_prediction = self.acne_model.predict(preprocessed_image)
            acne_condition = self.get_acne_label_name(acne_prediction)
            
            # Get skin type prediction
            skin_type_prediction = self.skin_type_model.predict(preprocessed_image)
            skin_type_label = np.argmax(skin_type_prediction, axis=1)[0]
            skin_type_name = self.get_skin_type_label_name(skin_type_label)
            
            # Save annotated image
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            output_path = Path(image_path).parent / f"annotated_{Path(image_path).name}"
            annotated_pil.save(output_path)
            
            return {
                "status": "success",
                "predictions": {
                    "acne_condition": acne_condition,
                    "skin_type": skin_type_name,
                    "detected_acne_types": detected_acne_types,
                },
                "annotated_image_path": f"/uploads/annotated_{Path(image_path).name}"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

