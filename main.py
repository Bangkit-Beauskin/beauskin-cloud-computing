from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

class ChatInput(BaseModel):
    message: str

app = FastAPI(title="BeauSkin API")

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
            "/analyze": "Analyze skin type and acne condition",
            "/chat": "Get chatbot response"
        }
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "File must be an image"}
            )
        
        # Save uploaded file
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the image
        result = await model_instance.analyze_image(file_path)
        
        # Add file URLs to result
        if result["status"] == "success":
            result["file_url"] = f"/uploads/{file.filename}"
        
        return result
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
    uvicorn.run(app, host="0.0.0.0", port=8000)