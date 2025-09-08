# backend/main.py - Final optimized version
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
import os
from dotenv import load_dotenv
from database import engine, get_db
import models
from config import upload_image
from face_api_service import face_service
from cache_service import cache_service  # Redis cache
from cdn_service import cdn_service      # CDN optimization
from typing import List, Dict, Optional
import numpy as np
import hashlib
import json
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from models import EmbeddingIndex

load_dotenv()

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Event Photo Search API - Optimized")

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Warm up cache on startup"""
    try:
        db = next(get_db())
        cache_service.warm_up_cache(db, limit=500)
        db.close()
    except:
        pass

# Helper functions
def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def calculate_query_hash(embedding: List[float], mode: str) -> str:
    """Create hash for cache key"""
    data = json.dumps({'emb': embedding[:10], 'mode': mode})
    return hashlib.md5(data.encode()).hexdigest()

async def vectorized_search(query_embedding: np.ndarray, 
                           embeddings_data: List[Dict],
                           threshold: float) -> List[Dict]:
    """Optimized vectorized search with Redis cache"""
    
    # Check Redis cache for embeddings
    if cache_service.is_available():
        embedding_ids = [e['id'] for e in embeddings_data]
        cached_embeddings = cache_service.batch_get_embeddings(embedding_ids)
        
        # Use cached embeddings where available
        for i, data in enumerate(embeddings_data):
            if data['id'] in cached_embeddings:
                data['embedding'] = cached_embeddings[data['id']]
    
    # Vectorized distance calculation
    embeddings_matrix = np.array([e['embedding'] for e in embeddings_data])
    distances = np.linalg.norm(embeddings_matrix - query_embedding, axis=1)
    
    # Filter by threshold
    matches = []
    for i, distance in enumerate(distances):
        if distance < threshold:
            matches.append({
                **embeddings_data[i],
                'distance': float(distance)
            })
    
    return matches

@app.get("/")
async def read_root():
    cache_stats = cache_service.get_cache_stats() if cache_service.is_available() else {}
    
    return {
        "message": "Event Photo Search API - Optimized",
        "features": {
            "redis_cache": cache_service.is_available(),
            "cdn_optimization": True,
            "background_processing": True,
            "vectorized_search": True
        },
        "cache_stats": cache_stats
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "status": "healthy",
        "database": db_status,
        "redis": cache_service.is_available(),
        "face_api": face_service.health_check()
    }

@app.post("/api/upload")
async def upload_image_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    event_id: Optional[int] = None,
    process_immediately: bool = Query(False),
    db: Session = Depends(get_db)
):
    try:
        contents = await file.read()
        
        # Check for duplicates
        file_hash = calculate_file_hash(contents)
        existing = db.query(models.Image).filter_by(file_hash=file_hash).first()
        
        if existing:
            return JSONResponse(
                status_code=409,
                content={
                    "success": False,
                    "message": "Image already exists",
                    "image_id": existing.id,
                    "url": existing.url,
                    "optimized_urls": cdn_service.get_responsive_urls(existing.url)
                }
            )
        
        # Upload to Cloudinary
        url = upload_image(contents)
        
        # Save image record
        db_image = models.Image(
            url=url,
            event_id=event_id,
            file_hash=file_hash,
            processed=1 if process_immediately else 0
        )
        db.add(db_image)
        db.commit()
        db.refresh(db_image)
        
        # Process faces
        if process_immediately:
            # Process immediately and wait
            faces = await process_faces_immediate(db_image.id, url, db)
            
            return {
                "success": True,
                "image_id": db_image.id,
                "url": url,
                "optimized_urls": cdn_service.get_responsive_urls(url),
                "faces_detected": len(faces),
                "processing": "completed"
            }
        else:
            # Process in background
            background_tasks.add_task(
                process_faces_background,
                db_image.id,
                url
            )
            
            return {
                "success": True,
                "image_id": db_image.id,
                "url": url,
                "optimized_urls": cdn_service.get_responsive_urls(url),
                "processing": "background"
            }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

async def process_faces_immediate(image_id: int, url: str, db: Session) -> List[Dict]:
    """Process faces immediately and return results"""
    try:
        faces = face_service.extract_faces(url)
        
        # Cache embeddings in Redis
        embeddings_to_cache = {}
        
        for face in faces:
            emb_hash = hashlib.md5(
                json.dumps(face['embedding'][:10]).encode()
            ).hexdigest()
            
            embedding = models.FaceEmbedding(
                image_id=image_id,
                embedding=face['embedding'],
                bbox=face.get('area'),
                embedding_hash=emb_hash,
                quality_score=face.get('confidence', 0.5)
            )
            db.add(embedding)
            db.flush()  # Get ID
            
            # Prepare for Redis cache
            embeddings_to_cache[embedding.id] = face['embedding']
        
        # Batch cache to Redis
        if cache_service.is_available():
            cache_service.batch_cache_embeddings(embeddings_to_cache)
        
        # Update image
        image = db.query(models.Image).filter_by(id=image_id).first()
        if image:
            image.processed = 2
            image.face_count = len(faces)
        
        db.commit()
        return faces
        
    except Exception as e:
        db.rollback()
        raise e

def process_faces_background(image_id: int, url: str):
    """Background task to process faces"""
    db = next(get_db())
    try:
        faces = face_service.extract_faces(url)
        embeddings_to_cache = {}
        
        for face in faces:
            emb_hash = hashlib.md5(
                json.dumps(face['embedding'][:10]).encode()
            ).hexdigest()
            
            embedding = models.FaceEmbedding(
                image_id=image_id,
                embedding=face['embedding'],
                bbox=face.get('area'),
                embedding_hash=emb_hash,
                quality_score=face.get('confidence', 0.5)
            )
            db.add(embedding)
            db.flush()
            
            embeddings_to_cache[embedding.id] = face['embedding']
        
        # Cache to Redis
        if cache_service.is_available():
            cache_service.batch_cache_embeddings(embeddings_to_cache)
        
        # Update image
        image = db.query(models.Image).filter_by(id=image_id).first()
        if image:
            image.processed = 2
            image.face_count = len(faces)
        
        db.commit()
        print(f"✓ Processed {len(faces)} faces for image {image_id}")
        
    except Exception as e:
        print(f"✗ Error processing image {image_id}: {e}")
        db.rollback()
    finally:
        db.close()

@app.post("/api/search")
async def search_faces(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    mode: str = Query("balanced", regex="^(strict|balanced|loose)$"),
    limit: int = Query(400, ge=1, le=600),
    include_thumbnails: bool = Query(False)
):
    try:
        contents = await file.read()
        query_url = upload_image(contents)
        
        # Extract face
        query_faces = face_service.extract_faces(query_url, return_all=False)
        
        if not query_faces:
            return {
                "success": False,
                "message": "No face found in query image",
                "query_url": query_url
            }
        
        query_embedding = np.array(query_faces[0]['embedding'])
        query_hash = calculate_query_hash(query_embedding.tolist(), mode)
        
        query_norm = float(np.linalg.norm(query_embedding))
        
        # Check Redis cache first
        # if cache_service.is_available():
        #     cached_result = cache_service.get_search_result(query_hash)
        #     if cached_result:
        #         # Add CDN optimized URLs
        #         for result in cached_result[:limit]:
        #             result['optimized_urls'] = cdn_service.get_responsive_urls(result['url'])
        #             if include_thumbnails and result.get('bbox'):
        #                 result['face_thumbnails'] = cdn_service.get_face_thumbnails(
        #                     result['url'], [result['bbox']]
        #                 )
                
        #         return {
        #             "success": True,
        #             "query_url": query_url,
        #             "query_thumbnail": cdn_service.get_optimized_url(query_url, 'thumbnail'),
        #             "results": cached_result[:limit],
        #             "from_cache": True,
        #             "mode": mode
        #         }
        
        # Get embeddings based on mode
        min_quality = 0.2 if mode == "loose" else 0.35 if mode == "balanced" else 0.45
        
        # Query with optimizations
        query = db.query(
    models.FaceEmbedding.id,
    models.FaceEmbedding.image_id,
    models.FaceEmbedding.embedding,
    models.FaceEmbedding.bbox,
    models.FaceEmbedding.quality_score,
    models.Image.url,
    models.Image.uploaded_at,
    models.EmbeddingIndex.norm
).join(
    models.Image
).join(
    models.EmbeddingIndex, 
    models.FaceEmbedding.id == models.EmbeddingIndex.face_embedding_id
).filter(
    and_(
        models.FaceEmbedding.quality_score >= min_quality,
        models.Image.processed == 2,
        # Pre-filter by norm range (rough similarity))
        models.EmbeddingIndex.norm.between(
            query_norm - 0.8, 
            query_norm + 0.8
        )
    )
)
        
        # Limit scope for strict mode
        if mode == "strict":
            query = query.order_by(desc(models.Image.uploaded_at)).limit(4000)
        else:
            query = query.limit(6000)
        
        results = query.all()
        
        # Prepare embeddings dataa
        embeddings_data = [
            {
                'id': r.id,
                'image_id': r.image_id,
                'embedding': r.embedding,
                'bbox': r.bbox,
                'quality': r.quality_score,
                'url': r.url,
                'uploaded_at': r.uploaded_at
            }
            for r in results
        ]
        
        # Calculate thresholds
        sample_distances = np.random.choice(
            len(embeddings_data), 
            min(200, len(embeddings_data)), 
            replace=False
        ).tolist() if embeddings_data else []
        
        if sample_distances:
            sample_embeddings = [embeddings_data[i]['embedding'] for i in sample_distances]
            distances = np.linalg.norm(
                np.array(sample_embeddings) - query_embedding, 
                axis=1
            ).tolist()
        else:
            distances = []
        
        strict_threshold, loose_threshold = face_service.calculate_adaptive_threshold(distances)
        threshold = strict_threshold if mode == "strict" else loose_threshold if mode == "loose" else strict_threshold * 1.2
        
        # Vectorized search
        matches = await vectorized_search(query_embedding, embeddings_data, threshold)
        
        # Group by image
        image_scores = {}
        for match in matches:
            img_id = match['image_id']
            if img_id not in image_scores:
                image_scores[img_id] = {
                    'url': match['url'],
                    'distances': [],
                    'bboxes': [],
                    'best_quality': 0,
                    'uploaded_at': match['uploaded_at']
                }
            image_scores[img_id]['distances'].append(match['distance'])
            image_scores[img_id]['bboxes'].append(match['bbox'])
            image_scores[img_id]['best_quality'] = max(
                image_scores[img_id]['best_quality'], 
                match['quality']
            )
        
        # Build final results
        final_results = []
        for img_id, data in image_scores.items():
            min_distance = min(data['distances'])
            avg_distance = np.mean(data['distances'])
            
            # Composite score
            composite_score = (
                min_distance * 0.4 + 
                avg_distance * 0.3 + 
                (1 / (len(data['distances']) + 1)) * 0.2 +
                (1 - data['best_quality']) * 0.1
            )
            
            result = {
                'image_id': img_id,
                'url': data['url'],
                'optimized_urls': cdn_service.get_responsive_urls(data['url']),
                'min_distance': float(min_distance),
                'avg_distance': float(avg_distance),
                'face_count': len(data['distances']),
                'composite_score': float(composite_score),
                'confidence': max(0, 1 - min_distance),
                'uploaded_at': data['uploaded_at'].isoformat()
            }
            
            # Add face thumbnails if requested
            if include_thumbnails and data['bboxes']:
                result['face_thumbnails'] = cdn_service.get_face_thumbnails(
                    data['url'], 
                    data['bboxes'][:3]  # Max 3 face thumbnails
                )
            
            final_results.append(result)
        
        # Sort and limit
        final_results.sort(key=lambda x: x['composite_score'])
        final_results = final_results[:limit]
        
        # Cache results in Redis
        if cache_service.is_available() and final_results:
            cache_service.cache_search_result(query_hash, final_results, ttl_minutes=30)
        
        return {
            "success": True,
            "query_url": query_url,
            "query_thumbnail": cdn_service.get_optimized_url(query_url, 'thumbnail'),
            "results": final_results,
            "total_matches": len(final_results),
            "thresholds": {
                "strict": strict_threshold,
                "balanced": (strict_threshold + loose_threshold) / 2,
                "loose": loose_threshold,
                "used": threshold
            },
            "mode": mode,
            "from_cache": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/images")
async def get_all_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),  
    event_id: Optional[int] = None,
    processed_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get images with proper pagination"""
    query = db.query(models.Image)
    
    if event_id:
        query = query.filter(models.Image.event_id == event_id)
    
    if processed_only:
        query = query.filter(models.Image.processed == 2)
    
    total = query.count()
    
    images = query.order_by(
        desc(models.Image.uploaded_at)
    ).offset(skip).limit(limit).all()
    
    return {
        "images": [
            {
                "id": img.id,
                "url": img.url,
                "uploaded_at": img.uploaded_at.isoformat() if img.uploaded_at else None,
                "face_count": getattr(img, 'face_count', 0),  # Safe access
                "event_id": img.event_id,
                "processed": getattr(img, 'processed', 2)  # Default to 2 if not exist
            } for img in images
        ],
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total
    }

@app.get("/api/image/{image_id}")
async def get_image_details(
    image_id: int,
    include_similar: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Get detailed image information with faces"""
    image = db.query(models.Image).filter_by(id=image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get faces
    faces = db.query(models.FaceEmbedding).filter_by(image_id=image_id).all()
    
    # Build response
    response = {
        "id": image.id,
        "url": image.url,
        "optimized_urls": cdn_service.get_responsive_urls(image.url),
        "placeholder": cdn_service.generate_placeholder(image.url),
        "uploaded_at": image.uploaded_at.isoformat(),
        "face_count": image.face_count,
        "event_id": image.event_id,
        "faces": [
            {
                "id": face.id,
                "bbox": face.bbox,
                "quality": face.quality_score,
                "thumbnail": cdn_service.get_face_thumbnails(
                    image.url, 
                    [face.bbox]
                )[0] if face.bbox else None
            }
            for face in faces
        ]
    }
    
    # Get similar images if requested
    if include_similar and cache_service.is_available():
        similar = cache_service.get_similar_images(image_id, limit=10)
        if similar:
            similar_images = db.query(models.Image).filter(
                models.Image.id.in_([s[0] for s in similar])
            ).all()
            
            response["similar_images"] = [
                {
                    "id": img.id,
                    "url": img.url,
                    "thumbnail": cdn_service.get_optimized_url(img.url, 'thumbnail'),
                    "similarity": next(s[1] for s in similar if s[0] == img.id)
                }
                for img in similar_images
            ]
    
    return response

@app.post("/api/batch-upload")
async def batch_upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    event_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Batch upload multiple images"""
    results = []
    
    for file in files[:50]:  # Max 50 files
        try:
            contents = await file.read()
            file_hash = calculate_file_hash(contents)
            
            # Check duplicate
            existing = db.query(models.Image).filter_by(file_hash=file_hash).first()
            if existing:
                results.append({
                    "filename": file.filename,
                    "status": "duplicate",
                    "image_id": existing.id,
                    "url": existing.url
                })
                continue
            
            # Upload
            url = upload_image(contents)
            
            # Save
            db_image = models.Image(
                url=url,
                event_id=event_id,
                file_hash=file_hash,
                processed=0
            )
            db.add(db_image)
            db.flush()
            
            # Queue for processing
            background_tasks.add_task(
                process_faces_background,
                db_image.id,
                url
            )
            
            results.append({
                "filename": file.filename,
                "status": "uploaded",
                "image_id": db_image.id,
                "url": url,
                "thumbnail": cdn_service.get_optimized_url(url, 'thumbnail')
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    db.commit()
    
    return {
        "success": True,
        "total": len(files),
        "uploaded": sum(1 for r in results if r["status"] == "uploaded"),
        "duplicates": sum(1 for r in results if r["status"] == "duplicate"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "results": results
    }

@app.delete("/api/cache/clear")
async def clear_cache(
    pattern: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Clear cache"""
    cleared = 0
    
    # Clear Redis cache
    if cache_service.is_available():
        cleared = cache_service.clear_cache(pattern)
    
    # Clear database cache
    if not pattern or pattern == "search":
        db.query(models.SearchCache).delete()
        db.commit()
    
    return {
        "success": True,
        "cleared": cleared,
        "message": f"Cleared cache with pattern: {pattern}" if pattern else "Cleared all cache"
    }

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get comprehensive statistics"""
    # Database stats
    total_images = db.query(models.Image).count()
    processed_images = db.query(models.Image).filter_by(processed=2).count()
    total_faces = db.query(models.FaceEmbedding).count()
    
    # Cache stats
    cache_stats = cache_service.get_cache_stats() if cache_service.is_available() else {}
    
    # Recent activity
    recent_uploads = db.query(
        func.date(models.Image.uploaded_at).label('date'),
        func.count(models.Image.id).label('count')
    ).group_by(
        func.date(models.Image.uploaded_at)
    ).order_by(
        desc('date')
    ).limit(7).all()
    
    return {
        "database": {
            "total_images": total_images,
            "processed_images": processed_images,
            "processing_images": total_images - processed_images,
            "total_faces": total_faces,
            "avg_faces_per_image": round(total_faces / processed_images, 2) if processed_images > 0 else 0
        },
        "cache": cache_stats,
        "recent_activity": [
            {"date": str(r.date), "uploads": r.count}
            for r in recent_uploads
        ]
    }

@app.get("/api/download/{image_id}")
async def get_download_url(
    image_id: int,
    quality: str = Query("full", regex="^(full|display|original)$"),
    db: Session = Depends(get_db)
):
    """Get download URL for image"""
    image = db.query(models.Image).filter_by(id=image_id).first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    if quality == "original":
        download_url = image.url
    else:
        download_url = cdn_service.get_optimized_url(image.url, quality)
    
    # Add download headers
    download_url = cdn_service.get_download_url(
        download_url, 
        f"event_photo_{image_id}.jpg"
    )
    
    return {
        "image_id": image_id,
        "download_url": download_url,
        "quality": quality
    }

@app.delete("/api/clear-all")
async def clear_all_data(db: Session = Depends(get_db)):
    """Xóa tất cả dữ liệu trong database"""
    try:
        # Clear cache first
        if cache_service.is_available():
            cache_service.clear_cache()
        
        # Delete all data
        db.query(models.EmbeddingIndex).delete()
        db.query(models.FaceEmbedding).delete()
        db.query(models.Image).delete()
        db.query(models.SearchCache).delete()
        
        db.commit()
        
        return {
            "success": True,
            "message": "Đã xóa tất cả dữ liệu thành công"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)