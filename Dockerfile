# Use the Python 3.9 slim base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy your application code and the service account key
COPY . /app
COPY /config/service-account-key.json /app/service-account-key.json

# Install necessary dependencies (including OpenCV, TensorFlow CPU, and other libraries)
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (use CPU-only TensorFlow version)
RUN pip install --upgrade pip
RUN pip install \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles \
    tensorflow-cpu \
    pillow \
    ultralytics \
    opencv-python-headless \
    numpy \
    scikit-learn \
    google-cloud-storage

# Set the environment variable for the Google Cloud service account key
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/service-account-key.json"

# Expose the port
ENV PORT=8080
EXPOSE 8080

# Run the application using Uvicorn with FastAPI
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
