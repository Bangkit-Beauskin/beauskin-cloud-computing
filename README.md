# beauskin-cloud-computing

# clone project 
git clone https://github.com/Bangkit-Beauskin/beauskin-cloud-computing

# installation


# ho to run project
python3 -m venv venv

source venv/bin/activate

pip install fastapi uvicorn python-multipart aiofiles tensorflow pillow ultralytics opencv-python-headless numpy tensorflow-cpu scikit-learn

uvicorn main:app --reload --host 0.0.0.0 --port 8000

deactivate