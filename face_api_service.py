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
        self.min_request_interval = 0.5  # Minimum 0.5s between requests
    
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
                    timeout=45  # Increase timeout
                )
                
                print(f"Face API response status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    faces = data.get("faces", [])
                    print(f"Faces found: {len(faces)}")
                    
                    # If no faces found and we haven't reached max retries, try again
                    if len(faces) == 0 and attempt < max_retries - 1:
                        print(f"No faces found, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    
                    return faces
                    
                elif response.status_code == 500 and attempt < max_retries - 1:
                    # Server error, retry
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
        
        # If all retries failed
        print(f"All retries failed for {image_url}")
        return []