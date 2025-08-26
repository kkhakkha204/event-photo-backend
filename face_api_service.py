import requests
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class FaceAPIService:
    def __init__(self):
        self.api_url = os.getenv("FACE_API_URL", "http://localhost:5000")
        print(f"Face API URL: {self.api_url}")
        
        # Setup session với retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def health_check(self) -> bool:
        """Check if face API service is healthy"""
        try:
            response = self.session.get(
                f"{self.api_url}/health",
                timeout=5
            )
            data = response.json()
            return data.get("status") == "healthy"
        except:
            return False
    
    def extract_faces(self, image_url: str, return_all: bool = True, retry_count: int = 0) -> List[Dict]:
        """Extract faces và embeddings từ image với retry logic"""
        max_retries = 3
        
        try:
            print(f"Calling face API with image: {image_url} (attempt {retry_count + 1})")
            
            # Check health trước khi gọi
            if retry_count == 0 and not self.health_check():
                print("Face API service unhealthy, waiting...")
                time.sleep(2)
            
            response = self.session.post(
                f"{self.api_url}/detect",
                json={
                    "image_url": image_url,
                    "return_all_faces": return_all
                },
                timeout=60  # Tăng timeout cho ảnh lớn
            )
            
            print(f"Face API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                faces = data.get("faces", [])
                print(f"Faces found: {len(faces)}")
                
                # Nếu không tìm thấy face và còn retry
                if len(faces) == 0 and retry_count < max_retries - 1:
                    print(f"No faces found, retrying in 2 seconds...")
                    time.sleep(2)
                    return self.extract_faces(image_url, return_all, retry_count + 1)
                
                return faces
                
            elif response.status_code in [502, 503, 504]:
                # Service có thể đang restart
                if retry_count < max_retries - 1:
                    wait_time = 3 * (retry_count + 1)
                    print(f"Service unavailable, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    return self.extract_faces(image_url, return_all, retry_count + 1)
                else:
                    print(f"Max retries reached. Face API error: {response.text}")
                    return []
            else:
                print(f"Face API error: {response.text}")
                return []
                
        except requests.exceptions.Timeout:
            print(f"Timeout error for image: {image_url}")
            if retry_count < max_retries - 1:
                time.sleep(3)
                return self.extract_faces(image_url, return_all, retry_count + 1)
            return []
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error: {e}")
            if retry_count < max_retries - 1:
                time.sleep(5)
                return self.extract_faces(image_url, return_all, retry_count + 1)
            return []
            
        except Exception as e:
            print(f"Unexpected error calling face API: {e}")
            return []
    
    def compare_faces(self, embedding1: List[float], embedding2: List[float]) -> float:
        """So sánh 2 face embeddings với Euclidean distance"""
        try:
            # Validate embeddings
            if not embedding1 or not embedding2:
                return 1.0  # Max distance
            
            if len(embedding1) != len(embedding2):
                print(f"Embedding size mismatch: {len(embedding1)} vs {len(embedding2)}")
                return 1.0
            
            # Calculate Euclidean distance
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
        
        # Filter out invalid distances
        valid_distances = [d for d in distances if 0 <= d <= 2]
        
        if not valid_distances:
            return 0.4, 0.5
        
        mean_dist = np.mean(valid_distances)
        std_dist = np.std(valid_distances)
        min_dist = np.min(valid_distances)
        
        # Calculate thresholds với bounds
        strict_threshold = min(0.35, max(0.2, mean_dist - std_dist))
        loose_threshold = min(0.6, max(0.4, mean_dist + std_dist * 0.5))
        
        return strict_threshold, loose_threshold
    
    def batch_extract_faces(self, image_urls: List[str], batch_size: int = 5) -> Dict[str, List[Dict]]:
        """Extract faces từ nhiều ảnh với batching"""
        results = {}
        
        for i in range(0, len(image_urls), batch_size):
            batch = image_urls[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_urls) + batch_size - 1)//batch_size}")
            
            for url in batch:
                faces = self.extract_faces(url)
                results[url] = faces
                
                # Small delay between requests
                time.sleep(0.5)
            
            # Longer delay between batches
            if i + batch_size < len(image_urls):
                time.sleep(2)
        
        return results

face_service = FaceAPIService()