from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
from database import engine, get_db
import models
from config import upload_image
from face_service import extract_faces

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
        faces = extract_faces(url)
        
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