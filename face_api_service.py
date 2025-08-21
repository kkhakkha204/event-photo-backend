import requests
import os
from typing import List, Dict

class FaceAPIService:
    def __init__(self):
        self.api_url = os.getenv("FACE_API_URL", "http://localhost:5000")
        print(f"Face API URL: {self.api_url}")
    
    def extract_faces(self, image_url: str) -> List[Dict]:
        """Extract faces và embeddings từ image"""
        try:
            print(f"Calling face API with image: {image_url}")
            response = requests.post(
                f"{self.api_url}/detect",
                json={"image_url": image_url},
                timeout=30  # Tăng timeout
            )
            
            print(f"Face API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Faces found: {len(data.get('faces', []))}")
                return data.get("faces", [])
            else:
                print(f"Face API error: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error calling face API: {e}")
            return []
    
    def compare_faces(self, embedding1: List[float], embedding2: List[float]) -> float:
        """So sánh 2 face embeddings"""
        diff = [a - b for a, b in zip(embedding1, embedding2)]
        distance = sum(x * x for x in diff) ** 0.5
        return distance

face_service = FaceAPIService()