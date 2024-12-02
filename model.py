# model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import asyncio
import random

class BeauSkinModel:
    def __init__(self):
        try:
            # Load responses dictionary and label encoder
            print("Loading chatbot files...")
            self.responses_dict = np.load('chatbot_responses.npy', allow_pickle=True).item()
            self.label_encoder = np.load('chatbot_label_encoder.npy', allow_pickle=True)
            
            # Load weights
            self.skin_type_model = tf.keras.models.load_model('skintypes_detection_model.h5')
            self.acne_model = tf.keras.models.load_model('acne_grade_model2.h5')
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
        """Preprocess image for prediction"""
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def get_skin_type_label(self, label):
        """Get skin type label name"""
        skin_type_labels = {0: 'dry', 1: 'normal', 2: 'oily'}
        return skin_type_labels.get(label, "Unknown")

    def get_acne_label(self, prediction):
        """Get acne condition label"""
        class_labels = {0: "Mild", 1: "Moderate", 2: "Normal", 3: "Severe"}
        predicted_index = np.argmax(prediction)
        return class_labels[predicted_index]

    async def analyze_image(self, file):
        try:
            # Read and preprocess image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = self.preprocess_image(image)

            # Get skin type prediction
            skin_type_pred = self.skin_type_model.predict(processed_image)
            skin_type_index = np.argmax(skin_type_pred[0])
            skin_type = self.get_skin_type_label(skin_type_index)
            skin_confidence = float(skin_type_pred[0][skin_type_index])

            # Get acne prediction
            acne_pred = self.acne_model.predict(processed_image)
            acne_condition = self.get_acne_label(acne_pred)
            acne_confidence = float(np.max(acne_pred))

            return {
                "status": "success",
                "predictions": {
                    "skin_type": {
                        "condition": skin_type,
                        "confidence": skin_confidence
                    },
                    "acne": {
                        "condition": acne_condition,
                        "confidence": acne_confidence
                    }
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_chatbot_response(self, text):
        """Get chatbot response based on input text"""
        try:
            # Get intent
            intent = self.get_intent(text)
            
            # Get responses for this intent
            responses = self.responses_dict[intent]
            
            # Select random response
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

    