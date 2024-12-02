# model.py
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import asyncio
import random
from ultralytics import YOLO
import cv2
import base64


class BeauSkinModel:
    def __init__(self, models_dir: str):
        try:
            self.models_dir = Path(models_dir)
            
            # Load responses dictionary and label encoder
            print("Loading chatbot files...")
            self.responses_dict = np.load(self.models_dir / 'chatbot_responses.npy', allow_pickle=True).item()
            self.label_encoder = np.load(self.models_dir / 'chatbot_label_encoder.npy', allow_pickle=True)
            
            # Load all three models
            self.acne_model = tf.keras.models.load_model(self.models_dir / 'acne_grade_model.h5')
            self.skin_type_model = tf.keras.models.load_model(self.models_dir / 'skintypes_detection_model.h5')
            self.acne_types_model = YOLO(str(self.models_dir / 'best_model.pt'))
            
            # Define fixed colors for YOLO labels
            self.label_colors = {
                'blackheads': (255, 0, 0),    # Red
                'dark spot': (0, 255, 0),     # Green
                'nodules': (0, 0, 255),       # Blue
                'papules': (255, 255, 0),     # Yellow
                'pustules': (0, 255, 255),    # Cyan
                'whiteheads': (255, 0, 255)   # Magenta
            }
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def get_intent(self, text):
        """Determine the intent of the input text"""
        text = text.lower().strip()
        
        if 'hello' in text or 'hi' in text:
            return 'greeting'
        elif 'bye' in text or 'goodbye' in text:
            return 'goodbye'
        elif 'thank' in text:
            return 'thank_you'
        elif 'dry' in text:
            return 'skin_treatment_dry'
        elif 'oily' in text:
            return 'skin_treatment_oily'
        elif 'acne' in text:
            return 'skin_treatment_acne'
        elif 'normal' in text:
            return 'skin_treatment_normal'
        elif 'sunscreen' in text or 'sun' in text:
            return 'sun_protection'
        elif 'product' in text:
            return 'product_recommendations'
        elif 'allergy' in text or 'reaction' in text:
            return 'allergic_reaction'
        elif 'routine' in text or 'use' in text:
            return 'product_usage'
        elif 'type' in text:
            return 'skin_type'
        else:
            return 'common_skin_issues'

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

    async def get_chatbot_response(self, text):
        """Get chatbot response based on input text"""
        try:
            intent = self.get_intent(text)
            responses = self.responses_dict[intent]
            response = random.choice(responses)
            
            return {
                "status": "success",
                "response": str(response),
                "intent": intent
            }
        except Exception as e:
            print(f"Chatbot error: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }