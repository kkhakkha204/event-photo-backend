import numpy as np
from deepface import DeepFace
import tempfile
import requests
from typing import List, Dict
import os

def download_image(url: str) -> str:
    """Download image từ URL về temp file"""
    response = requests.get(url)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name

def extract_faces(image_url: str) -> List[Dict]:
    """Extract face embeddings từ image"""
    temp_path = download_image(image_url)
    
    try:
        # Detect faces và extract embeddings
        # Dùng mtcnn thay vì opencv
        faces = DeepFace.represent(
            img_path=temp_path,
            model_name="Facenet512",
            detector_backend="mtcnn",
            enforce_detection=False
        )
        
        return faces
    except Exception as e:
        print(f"Error extracting faces: {e}")
        return []
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)