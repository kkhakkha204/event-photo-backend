from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
from database import engine, get_db
import models
from config import upload_image
from face_api_service import face_service
from typing import List, Dict
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from queue import Queue
import threading

load_dotenv()

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Event Photo Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:3000",  
        "https://event-photo-frontend.vercel.app/"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload queue system
class UploadQueue:
    def __init__(self):
        self.queue = Queue()
        self.processing = False
        self.results = {}
        self.executor = ThreadPoolExecutor(max_workers=1)  # Process one at a time
        
    def add_task(self, task_id: str, task_data: dict):
        self.queue.put((task_id, task_data))
        if not self.processing:
            self.start_processing()
    
    def start_processing(self):
        if not self.processing:
            self.processing = True
            thread = threading.Thread(target=self._process_queue)
            thread.daemon = True
            thread.start()
    
    def _process_queue(self):
        while not self.queue.empty():
            task_id, task_data = self.queue.get()
            try:
                # Add delay between processing to avoid overload
                time.sleep(1)
                result = self._process_image(task_data)
                self.results[task_id] = {"status": "completed", "result": result}
            except Exception as e:
                self.results[task_id] = {"status": "failed", "error": str(e)}
            finally:
                self.queue.task_done()
        self.processing = False
    
    def _process_image(self, task_data):
        contents = task_data['contents']
        db = task_data['db']
        
        # Upload to Cloudinary
        url = upload_image(contents)
        
        # Save to database
        db_image = models.Image(url=url)
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        # Extract faces with retry logic
        max_retries = 3
        faces = []
        for attempt in range(max_retries):
            try:
                faces = face_service.extract_faces(url)
                if faces or attempt == max_retries - 1:
                    break
                time.sleep(2)  # Wait before retry
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    raise
        
        # Save face embeddings
        for face in faces:
            embedding = models.FaceEmbedding(
                image_id=db_image.id,
                embedding=face['embedding'],
                bbox=face['area']
            )
            db.add(embedding)
        
        db.commit()
        
        return {
            "image_id": db_image.id,
            "url": url,
            "faces_detected": len(faces)
        }
    
    def get_status(self, task_id: str):
        if task_id in self.results:
            return self.results[task_id]
        elif any(task_id == tid for tid, _ in list(self.queue.queue)):
            position = next(i for i, (tid, _) in enumerate(list(self.queue.queue)) if tid == task_id)
            return {"status": "queued", "position": position + 1}
        else:
            return {"status": "not_found"}

upload_queue = UploadQueue()

@app.get("/")
def read_root():
    return {"message": "Event Photo Search API is running"}

@app.get("/health")
def health_check():
    face_api_health = face_service.check_health()
    return {
        "status": "healthy", 
        "database": "connected" if engine else "disconnected",
        "face_api": face_api_health,
        "queue_size": upload_queue.queue.qsize()
    }

@app.post("/api/upload")
async def upload_image_endpoint(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """Upload single image with queue system"""
    try:
        contents = await file.read()
        
        # Generate task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Add to queue
        upload_queue.add_task(task_id, {
            'contents': contents,
            'db': db
        })
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Image added to processing queue"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/upload/batch")
async def upload_batch_images(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Upload multiple images with queue system"""
    try:
        import uuid
        tasks = []
        
        for file in files[:20]:  # Limit to 20 files per batch
            contents = await file.read()
            task_id = str(uuid.uuid4())
            
            upload_queue.add_task(task_id, {
                'contents': contents,
                'db': db
            })
            
            tasks.append(task_id)
        
        return {
            "success": True,
            "task_ids": tasks,
            "message": f"{len(tasks)} images added to processing queue"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/upload/status/{task_id}")
async def get_upload_status(task_id: str):
    """Check status of upload task"""
    status = upload_queue.get_status(task_id)
    return status

@app.post("/api/upload/wait")
async def upload_image_wait(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and wait for processing (synchronous)"""
    try:
        contents = await file.read()
        
        # Process immediately without queue
        url = upload_image(contents)
        
        db_image = models.Image(url=url)
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        # Extract faces with timeout
        faces = []
        try:
            faces = await asyncio.wait_for(
                asyncio.to_thread(face_service.extract_faces, url),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            print(f"Face extraction timeout for {url}")
        
        for face in faces:
            embedding = models.FaceEmbedding(
                image_id=db_image.id,
                embedding=face['embedding'],
                bbox=face['area']
            )
            db.add(embedding)
        
        db.commit()
        
        return {
            "success": True,
            "image_id": db_image.id,
            "url": url,
            "faces_detected": len(faces)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    
@app.post("/api/search")
async def search_faces(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    mode: str = "balanced"
):
    try:
        contents = await file.read()
        query_url = upload_image(contents)
        
        # Extract faces with timeout
        query_faces = await asyncio.wait_for(
            asyncio.to_thread(face_service.extract_faces, query_url, False),
            timeout=30.0
        )
        
        if not query_faces:
            return {
                "success": False,
                "message": "Không tìm thấy khuôn mặt trong ảnh"
            }
        
        query_embedding = query_faces[0]['embedding']
        
        # Get all embeddings
        all_embeddings = db.query(models.FaceEmbedding).all()
        
        all_distances = []
        face_matches = []
        
        for face_emb in all_embeddings:
            distance = face_service.compare_faces(
                query_embedding, 
                face_emb.embedding
            )
            all_distances.append(distance)
            face_matches.append({
                'image_id': face_emb.image_id,
                'distance': distance,
                'bbox': face_emb.bbox,
                'embedding_id': face_emb.id
            })
        
        # Calculate adaptive thresholds
        strict_threshold, loose_threshold = face_service.calculate_adaptive_threshold(all_distances)
        
        if mode == "strict":
            threshold = strict_threshold
        elif mode == "loose":
            threshold = loose_threshold
        else:  
            threshold = (strict_threshold + loose_threshold) / 2
        
        # Filter matches
        matches = [m for m in face_matches if m['distance'] < threshold]
        
        # Group by image
        image_scores = {}
        for match in matches:
            img_id = match['image_id']
            if img_id not in image_scores:
                image_scores[img_id] = {
                    'distances': [],
                    'count': 0
                }
            image_scores[img_id]['distances'].append(match['distance'])
            image_scores[img_id]['count'] += 1
        
        # Calculate final scores
        results = []
        for img_id, scores in image_scores.items():
            min_distance = min(scores['distances'])
            avg_distance = np.mean(scores['distances'])
            face_count = scores['count']
            
            composite_score = (min_distance * 0.5) + (avg_distance * 0.3) + (1 / (face_count + 1) * 0.2)
            
            image = db.query(models.Image).filter(models.Image.id == img_id).first()
            if image:
                results.append({
                    'image_id': img_id,
                    'url': image.url,
                    'min_distance': min_distance,
                    'avg_distance': avg_distance,
                    'face_count': face_count,
                    'composite_score': composite_score,
                    'confidence': max(0, 1 - min_distance)
                })
        
        results.sort(key=lambda x: x['composite_score'])
        
        return {
            "success": True,
            "query_url": query_url,
            "results": results[:50],  # Limit results
            "total_images": len(results),
            "thresholds": {
                "strict": strict_threshold,
                "balanced": threshold,
                "loose": loose_threshold,
                "used": threshold
            },
            "mode": mode
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Face detection timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/images")
async def get_all_images(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    images = db.query(models.Image).offset(skip).limit(limit).all()
    total = db.query(models.Image).count()
    
    return {
        "images": [
            {
                "id": img.id,
                "url": img.url,
                "uploaded_at": img.uploaded_at.isoformat()
            } for img in images
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }

@app.delete("/api/images/{image_id}")
async def delete_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """Delete image and its embeddings"""
    try:
        # Delete embeddings first
        db.query(models.FaceEmbedding).filter(
            models.FaceEmbedding.image_id == image_id
        ).delete()
        
        # Delete image
        db.query(models.Image).filter(
            models.Image.id == image_id
        ).delete()
        
        db.commit()
        
        return {"success": True, "message": "Image deleted"}
    except Exception as e:
        db.rollback()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)