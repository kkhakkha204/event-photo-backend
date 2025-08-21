import requests
import base64
from typing import List, Dict
import numpy as np

# Sử dụng Face++ API (free tier)
FACE_API_KEY = "556Rpt7qg3825phuY15aq0htCrFM8p9N"
FACE_API_SECRET = "iXLOuqDgyNjnwaZmOkvqGv0gp3GYBFuu"

def extract_faces_simple(image_url: str) -> List[Dict]:
    """
    Extract face từ image dùng simple detection
    Tạm thời return empty để test flow
    """
    # TODO: Implement với Face++ API hoặc service khác
    return [{
        'embedding': [0.1] * 512,  # Fake embedding
        'area': {'x': 0, 'y': 0, 'w': 100, 'h': 100}
    }]