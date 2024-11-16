# main.py
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn
from model import BeauSkinModel

app = FastAPI(title="BeauSkin API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
model_instance = BeauSkinModel()

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

        result = await model_instance.analyze_image(file)
        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/chat")
async def chat(message: str = Body(..., embed=True)):
    try:
        result = await model_instance.get_chatbot_response(message)
        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)