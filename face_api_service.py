import requests
import base64
import json
from typing import List, Dict
import tempfile
import os

class FaceAPIService:
    def __init__(self):
        # Sử dụng face-api.js qua HTTP service
        self.api_url = os.getenv("FACE_API_URL", "http://localhost:5000")
    
    def extract_faces(self, image_url: str) -> List[Dict]:
        """Extract faces và embeddings từ image"""
        try:
            response = requests.post(
                f"{self.api_url}/detect",
                json={"image_url": image_url}
            )
            
            if response.status_code == 200:
                return response.json()["faces"]
            else:
                print(f"Face API error: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error calling face API: {e}")
            return []
    
    def compare_faces(self, embedding1: List[float], embedding2: List[float]) -> float:
        """So sánh 2 face embeddings"""
        # Euclidean distance
        diff = [a - b for a, b in zip(embedding1, embedding2)]
        distance = sum(x * x for x in diff) ** 0.5
        return distance

face_service = FaceAPIService()