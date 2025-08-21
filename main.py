from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
from database import engine, get_db
import models
from config import upload_image
from face_api_service import face_service
from typing import List
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
    threshold: float = 0.45  # Giảm threshold xuống 0.4
):
    try:
        # Upload ảnh query lên Cloudinary
        contents = await file.read()
        query_url = upload_image(contents)
        
        # Extract face từ ảnh query
        query_faces = face_service.extract_faces(query_url)
        
        if not query_faces:
            return {
                "success": False,
                "message": "Không tìm thấy khuôn mặt trong ảnh"
            }
        
        # Lấy embedding của khuôn mặt đầu tiên
        query_embedding = query_faces[0]['embedding']
        
        # Lấy tất cả embeddings từ database
        all_embeddings = db.query(models.FaceEmbedding).all()
        
        # Tìm matches với distance score
        matches = []
        for face_emb in all_embeddings:
            distance = face_service.compare_faces(
                query_embedding, 
                face_emb.embedding
            )
            
            if distance < threshold:
                matches.append({
                    'image_id': face_emb.image_id,
                    'distance': distance,
                    'bbox': face_emb.bbox,
                    'confidence': 1 - (distance / threshold)  # Confidence score
                })
        
        # Group by image và lấy thông tin ảnh
        image_ids = list(set([m['image_id'] for m in matches]))
        images = db.query(models.Image).filter(
            models.Image.id.in_(image_ids)
        ).all()
        
        # Format response với confidence score
        results = []
        for img in images:
            img_matches = [m for m in matches if m['image_id'] == img.id]
            # Tính average confidence cho image
            avg_confidence = sum(m['confidence'] for m in img_matches) / len(img_matches)
            results.append({
                'image_id': img.id,
                'url': img.url,
                'matches': len(img_matches),
                'confidence': avg_confidence,
                'faces': img_matches
            })
        
        # Sort by confidence score thay vì số lượng matches
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "success": True,
            "query_url": query_url,
            "results": results,
            "total_images": len(results),
            "threshold_used": threshold
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