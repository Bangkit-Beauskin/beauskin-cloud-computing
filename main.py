from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google.cloud import storage
from typing import List
from model import BeauSkinModel
import os
from pathlib import Path

# Create necessary folders
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/secrets/service-account"


class ChatInput(BaseModel):
    message: str

class MultiImageAnalysis(BaseModel):
    front_image: str
    left_image: str
    right_image: str

app = FastAPI(title="BeauSkin API")
storage_client = storage.Client()

bucket_name = "beauskin-main-bucket"
bucket = storage_client.bucket(bucket_name)

# Mount the uploads directory for static file serving
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing BeauSkin model...")
model_instance = BeauSkinModel(models_dir=str(MODELS_DIR))

@app.get("/")
async def index():
    return {
        "status": "success",
        "message": "BeauSkin API Ready",
        "endpoints": {
            "/analyze": "Analyze skin type and acne condition with multiple images",
            "/chat": "Get chatbot response"
        }
    }

@app.post("/analyze")
async def analyze(
    front_image: UploadFile = File(...),
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
):
    try:
        # Validate file types
        for image in [front_image, left_image, right_image]:
            if not image.content_type.startswith("image/"):
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"File {image.filename} must be an image"}
                )
        
        # Save and process each image
        results = {}
        image_files = {
            "front": front_image,
            "left": left_image,
            "right": right_image
        }
        
        for position, file in image_files.items():
            # Create position-specific subfolder
            position_dir = UPLOADS_DIR / position
            position_dir.mkdir(exist_ok=True)
            
            # Generate unique filename using original name
            original_filename = file.filename
            file_path = position_dir / original_filename
            
            # Save uploaded file locally
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process the image - note that model.analyze_image will save annotated image
            # in the same directory as the input image
            result = await model_instance.analyze_image(str(file_path))
            
            if result["status"] == "success":
                # Save annotated image to GCS
                annotated_filename = f"annotated_{original_filename}"
                annotated_file_path = position_dir / annotated_filename
                annotated_image_url = f"/uploads/{position}/{annotated_filename}"

                # Upload annotated image to GCS
                blob = bucket.blob(f"{position}/{annotated_filename}")
                with open(annotated_file_path, "rb") as annotated_image_file:
                    blob.upload_from_file(annotated_image_file)
                
                # Update result with the GCS URL for the annotated image
                result["file_url"] = f"/uploads/{position}/{original_filename}"
                result["annotated_image_path"] = f"https://storage.googleapis.com/{bucket_name}/{position}/{annotated_filename}"
                
                results[position] = result
            else:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"Error processing {position} image: {result['message']}"}
                )
        
        # Aggregate results
        combined_result = {
            "status": "success",
            "predictions": {
                "front": results["front"]["predictions"],
                "left": results["left"]["predictions"],
                "right": results["right"]["predictions"]
            },
            "original_images": {
                "front": results["front"]["file_url"],
                "left": results["left"]["file_url"],
                "right": results["right"]["file_url"]
            },
            "annotated_images": {
                "front": results["front"]["annotated_image_path"],
                "left": results["left"]["annotated_image_path"],
                "right": results["right"]["annotated_image_path"]
            }
        }
        
        return combined_result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/chat")
async def chat(input_data: ChatInput):
    try:
        result = await model_instance.get_chatbot_response(input_data.message)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)