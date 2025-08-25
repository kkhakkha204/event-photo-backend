import requests
import os
from typing import List, Dict, Tuple
import numpy as np

class FaceAPIService:
    def __init__(self):
        self.api_url = os.getenv("FACE_API_URL", "http://localhost:5000")
        print(f"Face API URL: {self.api_url}")
    
    def extract_faces(self, image_url: str, return_all: bool = True) -> List[Dict]:
        """Extract faces và embeddings từ image"""
        try:
            print(f"Calling face API with image: {image_url}")
            response = requests.post(
                f"{self.api_url}/detect",
                json={
                    "image_url": image_url,
                    "return_all_faces": return_all
                },
                timeout=30
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
        """So sánh 2 face embeddings với Euclidean distance"""
        diff = [a - b for a, b in zip(embedding1, embedding2)]
        distance = sum(x * x for x in diff) ** 0.5
        return distance
    
    def calculate_adaptive_threshold(self, distances: List[float]) -> Tuple[float, float]:
        """Tính threshold động dựa trên distribution của distances"""
        if not distances:
            return 0.4, 0.5
        
        # Tính statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        
        # Adaptive thresholds
        strict_threshold = min(0.35, mean_dist - std_dist)
        loose_threshold = min(0.5, mean_dist)
        
        return max(0.25, strict_threshold), min(0.6, loose_threshold)

face_service = FaceAPIService()