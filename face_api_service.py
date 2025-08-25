import requests
import os
from typing import List, Dict, Tuple
import numpy as np
import time

class FaceAPIService:
    def __init__(self):
        self.api_url = os.getenv("FACE_API_URL", "http://localhost:5000")
        print(f"Face API URL: {self.api_url}")
        self.last_request_time = 0
        self.min_request_interval = 0.5
    
    def _rate_limit(self):
        """Simple rate limiting to avoid overwhelming the service"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def check_health(self) -> Dict:
        """Check health of Face API service"""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}
    
    def extract_faces(self, image_url: str, return_all: bool = True) -> List[Dict]:
        """Extract faces và embeddings từ image với retry logic"""
        self._rate_limit()
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Calling face API (attempt {attempt + 1}/{max_retries}): {image_url}")
                
                response = requests.post(
                    f"{self.api_url}/detect",
                    json={
                        "image_url": image_url,
                        "return_all_faces": return_all
                    },
                    timeout=45
                )
                
                print(f"Face API response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    faces = data.get("faces", [])
                    print(f"Faces found: {len(faces)}")
                    
                    if len(faces) == 0 and attempt < max_retries - 1:
                        print(f"No faces found, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    
                    return faces
                    
                elif response.status_code == 500 and attempt < max_retries - 1:
                    print(f"Server error, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                    
                else:
                    print(f"Face API error: {response.text}")
                    return []
                    
            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return []
                
            except Exception as e:
                print(f"Error calling face API (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return []
        
        print(f"All retries failed for {image_url}")
        return []
    
    def compare_faces(self, embedding1: List[float], embedding2: List[float]) -> float:
        """So sánh 2 face embeddings với Euclidean distance"""
        try:
            if len(embedding1) != len(embedding2):
                print(f"Warning: Embedding size mismatch: {len(embedding1)} vs {len(embedding2)}")
                return 1.0
            
            diff = [a - b for a, b in zip(embedding1, embedding2)]
            distance = sum(x * x for x in diff) ** 0.5
            return distance
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return 1.0
    
    def calculate_adaptive_threshold(self, distances: List[float]) -> Tuple[float, float]:
        """Tính threshold động dựa trên distribution của distances"""
        if not distances:
            return 0.4, 0.5
        
        valid_distances = [d for d in distances if d <= 1.0]
        if not valid_distances:
            return 0.4, 0.5
        
        mean_dist = np.mean(valid_distances)
        std_dist = np.std(valid_distances)
        min_dist = np.min(valid_distances)
        percentile_25 = np.percentile(valid_distances, 25)
        percentile_75 = np.percentile(valid_distances, 75)
        
        strict_threshold = min(0.35, min_dist + std_dist * 0.5)
        loose_threshold = min(0.55, percentile_75)
        
        strict_threshold = max(0.25, strict_threshold)
        loose_threshold = min(0.6, max(strict_threshold + 0.1, loose_threshold))
        
        print(f"Threshold calculation - Min: {min_dist:.3f}, Mean: {mean_dist:.3f}, "
              f"Strict: {strict_threshold:.3f}, Loose: {loose_threshold:.3f}")
        
        return strict_threshold, loose_threshold
    
    def cleanup_face_api(self):
        """Request cleanup on Face API service"""
        try:
            response = requests.post(
                f"{self.api_url}/cleanup",
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                print(f"Face API cleanup: {result}")
                return result
        except Exception as e:
            print(f"Error calling cleanup: {e}")
        return None

# Khởi tạo instance
face_service = FaceAPIService()