# model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io

class BeauSkinModel:
    def __init__(self):
        try:
            # Load all models
            self.skin_type_model = tf.keras.models.load_model('model/weights/skintypes_detection_model.h5')
            self.acne_model = tf.keras.models.load_model('model/weights/acne_grade_model2.h5')
            self.chatbot_model = tf.keras.models.load_model('model/weights/chatbot_model.h5')
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {str(e)}")

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
        try:
            # Add your chatbot logic here
            # This is a placeholder - adjust according to your chatbot model's requirements
            response = self.chatbot_model.predict([text])
            return {
                "status": "success",
                "response": response
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}