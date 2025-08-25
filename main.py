from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
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

load_dotenv()

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Event Photo Search API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "http://localhost:3000",  # Local development
        "https://event-photo-frontend.vercel.app/"  # Thay bằng URL Vercel của bạn
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Event Photo Search API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "database": "connected" if engine else "disconnected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/api/upload")
async def upload_image_endpoint(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    try:
        # Upload to Cloudinary
        contents = await file.read()
        url = upload_image(contents)
        
        # Save to database
        db_image = models.Image(url=url)
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        # Extract faces
        faces = face_service.extract_faces(url)
        
        # Save embeddings
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
    mode: str = "balanced"  # strict, balanced, loose
):
    try:
        # Upload ảnh query
        contents = await file.read()
        query_url = upload_image(contents)
        
        # Extract faces từ ảnh query
        query_faces = face_service.extract_faces(query_url, return_all=False)
        
        if not query_faces:
            return {
                "success": False,
                "message": "Không tìm thấy khuôn mặt trong ảnh"
            }
        
        query_embedding = query_faces[0]['embedding']
        
        # Lấy tất cả embeddings
        all_embeddings = db.query(models.FaceEmbedding).all()
        
        # Tính distances cho tất cả faces
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
        
        # Tính adaptive thresholds
        strict_threshold, loose_threshold = face_service.calculate_adaptive_threshold(all_distances)
        
        # Chọn threshold theo mode
        if mode == "strict":
            threshold = strict_threshold
        elif mode == "loose":
            threshold = loose_threshold
        else:  # balanced
            threshold = (strict_threshold + loose_threshold) / 2
        
        # Filter matches
        matches = [m for m in face_matches if m['distance'] < threshold]
        
        # Ranking algorithm
        # 1. Group by image
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
        
        # 2. Calculate composite score for each image
        results = []
        for img_id, scores in image_scores.items():
            # Factors: min distance, average distance, face count
            min_distance = min(scores['distances'])
            avg_distance = np.mean(scores['distances'])
            face_count = scores['count']
            
            # Composite score (lower is better)
            composite_score = (min_distance * 0.5) + (avg_distance * 0.3) + (1 / (face_count + 1) * 0.2)
            
            # Get image info
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
        
        # Sort by composite score
        results.sort(key=lambda x: x['composite_score'])
        
        return {
            "success": True,
            "query_url": query_url,
            "results": results,
            "total_images": len(results),
            "thresholds": {
                "strict": strict_threshold,
                "balanced": threshold,
                "loose": loose_threshold,
                "used": threshold
            },
            "mode": mode
        }
        
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