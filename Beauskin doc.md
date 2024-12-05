# BeauSkin API Documentation

BeauSkin API is an AI-powered skin analysis service that provides automated assessment of skin conditions, acne detection, and skin type classification using machine learning models.

## Table of Contents
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Models](#models)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Bangkit-Beauskin/beauskin-cloud-computing
cd beauskin-cloud-computing
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/Mac
# OR
.\venv\Scripts\activate  # For Windows
```

3. Install required dependencies:
```bash
pip install fastapi uvicorn python-multipart aiofiles tensorflow pillow ultralytics opencv-python-headless numpy tensorflow-cpu scikit-learn
```

4. Place your trained models in the `models` directory:
```
models/
├── acne_grade_model.h5
├── skintypes_detection_model.h5
├── best_model.pt
├── chatbot_model.h5
└── skin_treatment.json
```

## Running the API

Start the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Multi-Image Analysis Endpoint

**Endpoint**: `/analyze`
**Method**: POST
**Content-Type**: multipart/form-data

Analyzes three images of the face (front, left, and right views) for skin conditions.

**Request Parameters**:
- `front_image`: Front view image file (required)
- `left_image`: Left view image file (required)
- `right_image`: Right view image file (required)

**Response Format**:
```json
{
  "status": "success",
  "predictions": {
    "front": {
      "acne_condition": "string",
      "skin_type": "string",
      "detected_acne_types": ["string"]
    },
    "left": {
      "acne_condition": "string",
      "skin_type": "string",
      "detected_acne_types": ["string"]
    },
    "right": {
      "acne_condition": "string",
      "skin_type": "string",
      "detected_acne_types": ["string"]
    }
  },
  "original_images": {
    "front": "string (URL)",
    "left": "string (URL)",
    "right": "string (URL)"
  },
  "annotated_images": {
    "front": "string (URL)",
    "left": "string (URL)",
    "right": "string (URL)"
  }
}
```

### 2. Chatbot Endpoint

**Endpoint**: `/chat`
**Method**: POST
**Content-Type**: application/json

Provides skin care advice and recommendations through a chatbot interface.

**Request Body**:
```json
{
  "message": "string"
}
```

**Response Format**:
```json
{
  "status": "success",
  "response": "string",
  "intent": "string"
}
```

## Models

The API uses several machine learning models:

1. **Acne Grade Model** (`acne_grade_model.h5`)
   - Classifies acne severity: Mild, Moderate, Severe, or No acne

2. **Skin Type Model** (`skintypes_detection_model.h5`)
   - Classifies skin type: Dry, Normal, or Oily

3. **Acne Types Detection Model** (`best_model.pt`)
   - YOLO model for detecting specific acne types:
     - Blackheads
     - Dark spots
     - Nodules
     - Papules
     - Pustules
     - Whiteheads

4. **Chatbot Model** (`chatbot_model.h5`)
   - Provides skin care advice and recommendations

5. **Skin Treatment Data** (`skin_treatment.json`)
   - Contains treatment recommendations and responses for the chatbot

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid input (e.g., non-image files)
- `500 Internal Server Error`: Server-side processing errors

Error response format:
```json
{
  "status": "error",
  "message": "Error description"
}
```

## Examples

### Using cURL

1. Multi-image analysis:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "front_image=@front.jpg" \
  -F "left_image=@left.jpg" \
  -F "right_image=@right.jpg"
```

2. Chatbot interaction:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What treatment is good for oily skin?"}'
```

### Using Python Requests

```python
import requests

# Multi-image analysis
files = {
    'front_image': open('front.jpg', 'rb'),
    'left_image': open('left.jpg', 'rb'),
    'right_image': open('right.jpg', 'rb')
}
response = requests.post('http://localhost:8000/analyze', files=files)
print(response.json())

# Chatbot
chat_response = requests.post(
    'http://localhost:8000/chat',
    json={'message': 'What treatment is good for oily skin?'}
)
print(chat_response.json())
```

## Troubleshooting

If you encounter any issues:

1. Ensure all required models are present in the `models` directory
2. Verify that the virtual environment is activated
3. Check if all dependencies are installed correctly
4. Ensure the required ports are not in use
5. Check the uploads directory has proper write permissions

For any additional issues, please open an issue in the GitHub repository.