# backend/optimized_main.py - API Optimization
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func, text
import asyncio
import aiofiles
from typing import List, Dict, Optional, AsyncGenerator
import time
import hashlib
import json
from datetime import datetime, timedelta

from database import engine, get_db
import models
from config import upload_image
from face_api_service import face_service  
from cache_service import cache_service
from cdn_service import cdn_service
from processing_pipeline import processing_pipeline

app = FastAPI(
    title="Event Photo Search API - Optimized",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware stack
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600  # Cache preflight for 1 hour
)

# Connection pooling and DB optimizations
from sqlalchemy.pool import QueuePool
engine.pool = QueuePool(
    engine.pool._creator,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300
)

# Response models for better serialization
from pydantic import BaseModel
from typing import Union

class ImageResponse(BaseModel):
    id: int
    url: str
    uploaded_at: Optional[str]
    face_count: int
    event_id: Optional[int]
    processed: int
    optimized_urls: Optional[Dict[str, str]] = None

class SearchResponse(BaseModel):
    success: bool
    query_url: str
    results: List[Dict]
    total_matches: int
    mode: str
    from_cache: bool = False
    processing_time: float

# Request/Response optimization
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Optimized database queries with proper indexing
class OptimizedQueries:
    @staticmethod
    def get_images_optimized(
        db: Session, 
        skip: int = 0, 
        limit: int = 20,
        event_id: Optional[int] = None,
        processed_only: bool = True
    ):
        """Optimized image query with proper indexing"""
        query = db.query(
            models.Image.id,
            models.Image.url,
            models.Image.uploaded_at,
            models.Image.face_count,
            models.Image.event_id,
            models.Image.processed
        )
        
        # Use indexes efficiently
        if processed_only:
            query = query.filter(models.Image.processed == 2)
        if event_id:
            query = query.filter(models.Image.event_id == event_id)
        
        # Use index on uploaded_at for ordering
        total = query.count()
        images = query.order_by(
            desc(models.Image.uploaded_at)
        ).offset(skip).limit(limit).all()
        
        return images, total
    
    @staticmethod 
    def get_embeddings_for_search(
        db: Session,
        min_quality: float = 0.4,
        limit: int = 10000
    ):
        """Optimized embedding query for search"""
        return db.query(
            models.FaceEmbedding.id,
            models.FaceEmbedding.image_id,
            models.FaceEmbedding.embedding,
            models.FaceEmbedding.bbox,
            models.FaceEmbedding.quality_score,
            models.Image.url,
            models.Image.uploaded_at
        ).select_from(
            models.FaceEmbedding
        ).join(
            models.Image,
            models.FaceEmbedding.image_id == models.Image.id
        ).filter(
            and_(
                models.FaceEmbedding.quality_score >= min_quality,
                models.Image.processed == 2
            )
        ).order_by(
            desc(models.Image.uploaded_at)
        ).limit(limit).all()

# Streaming responses for large datasets
async def stream_images(images: List, include_cdn: bool = False) -> AsyncGenerator[str, None]:
    """Stream images as NDJSON for better memory usage"""
    yield '{"images": ['
    
    for i, img in enumerate(images):
        if i > 0:
            yield ","
        
        image_data = {
            "id": img.id,
            "url": img.url,
            "uploaded_at": img.uploaded_at.isoformat() if img.uploaded_at else None,
            "face_count": getattr(img, 'face_count', 0),
            "event_id": img.event_id,
            "processed": getattr(img, 'processed', 2)
        }
        
        if include_cdn:
            image_data["optimized_urls"] = cdn_service.get_responsive_urls(img.url)
        
        yield json.dumps(image_data)
        
        # Yield control for other tasks
        if i % 10 == 0:
            await asyncio.sleep(0)
    
    yield ']}'

# Cached endpoints
@app.get("/api/images/cached")
@cache(expire=300)  # 5 minutes cache
async def get_cached_images(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=50),
    event_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Cached image list with CDN URLs"""
    images, total = OptimizedQueries.get_images_optimized(
        db, skip, limit, event_id, processed_only=True
    )
    
    # Batch optimize URLs
    urls = [img.url for img in images]
    thumbnails = cdn_service.batch_optimize_urls(urls, 'thumbnail')
    
    return {
        "images": [
            {
                "id": img.id,
                "url": img.url,
                "thumbnail": thumbnails[i],
                "uploaded_at": img.uploaded_at.isoformat() if img.uploaded_at else None,
                "face_count": getattr(img, 'face_count', 0),
                "event_id": img.event_id
            }
            for i, img in enumerate(images)
        ],
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total
    }

@app.get("/api/images/stream")
async def stream_images_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    include_cdn: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Stream large image datasets"""
    images, _ = OptimizedQueries.get_images_optimized(db, skip, limit)
    
    return StreamingResponse(
        stream_images(images, include_cdn),
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Total-Count": str(len(images))
        }
    )

# Optimized search endpoint
@app.post("/api/search/optimized")
async def optimized_search(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    mode: str = Query("balanced", regex="^(strict|balanced|loose)$"),
    limit: int = Query(20, ge=1, le=100),
    quality_threshold: Optional[float] = Query(None, ge=0.1, le=1.0)
):
    """Optimized search with better caching and vectorization"""
    start_time = time.time()
    
    try:
        # Read file efficiently
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Generate cache key
        file_hash = hashlib.md5(contents).hexdigest()
        cache_key = f"search:{file_hash}:{mode}:{limit}"
        
        # Check cache first
        if cache_service.is_available():
            cached_result = cache_service.get_search_result(cache_key)
            if cached_result:
                return SearchResponse(
                    success=True,
                    query_url="cached",
                    results=cached_result[:limit],
                    total_matches=len(cached_result),
                    mode=mode,
                    from_cache=True,
                    processing_time=time.time() - start_time
                )
        
        # Upload and extract faces
        query_url = upload_image(contents)
        query_faces = face_service.extract_faces(query_url, return_all=False)
        
        if not query_faces:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "message": "No face detected in query image",
                    "query_url": query_url
                }
            )
        
        import numpy as np
        query_embedding = np.array(query_faces[0]['embedding'], dtype=np.float32)
        
        # Dynamic quality threshold
        if quality_threshold is None:
            quality_threshold = 0.6 if mode == "strict" else 0.4 if mode == "balanced" else 0.3
        
        # Optimized database query
        embeddings_data = OptimizedQueries.get_embeddings_for_search(
            db, quality_threshold, 15000 if mode == "loose" else 10000
        )
        
        if not embeddings_data:
            return SearchResponse(
                success=True,
                query_url=query_url,
                results=[],
                total_matches=0,
                mode=mode,
                from_cache=False,
                processing_time=time.time() - start_time
            )
        
        # Vectorized search with optimized numpy operations
        embeddings_matrix = np.array([
            emb.embedding for emb in embeddings_data
        ], dtype=np.float32)
        
        # Compute distances using optimized numpy
        distances = np.linalg.norm(
            embeddings_matrix - query_embedding, 
            axis=1, 
            ord=2
        )
        
        # Dynamic threshold calculation
        threshold_percentiles = {
            "strict": 15,
            "balanced": 25, 
            "loose": 35
        }
        threshold = np.percentile(distances, threshold_percentiles[mode])
        threshold = max(0.3, min(0.8, threshold))  # Bounds
        
        # Filter matches
        mask = distances < threshold
        valid_indices = np.where(mask)[0]
        
        # Build results efficiently
        image_matches = {}
        for idx in valid_indices:
            emb_data = embeddings_data[idx]
            distance = float(distances[idx])
            
            img_id = emb_data.image_id
            if img_id not in image_matches:
                image_matches[img_id] = {
                    'url': emb_data.url,
                    'distances': [],
                    'bboxes': [],
                    'uploaded_at': emb_data.uploaded_at
                }
            
            image_matches[img_id]['distances'].append(distance)
            if emb_data.bbox:
                image_matches[img_id]['bboxes'].append(emb_data.bbox)
        
        # Rank and format results
        final_results = []
        for img_id, data in image_matches.items():
            min_dist = min(data['distances'])
            avg_dist = sum(data['distances']) / len(data['distances'])
            
            final_results.append({
                'image_id': img_id,
                'url': data['url'],
                'thumbnail': cdn_service.get_optimized_url(data['url'], 'thumbnail'),
                'min_distance': min_dist,
                'avg_distance': avg_dist,
                'face_count': len(data['distances']),
                'confidence': max(0, 1 - min_dist),
                'uploaded_at': data['uploaded_at'].isoformat()
            })
        
        # Sort by composite score
        final_results.sort(key=lambda x: x['min_distance'])
        final_results = final_results[:limit]
        
        # Cache results
        if cache_service.is_available() and final_results:
            cache_service.cache_search_result(cache_key, final_results, ttl_minutes=60)
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            success=True,
            query_url=query_url,
            results=final_results,
            total_matches=len(final_results),
            mode=mode,
            from_cache=False,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch upload with progress tracking
@app.post("/api/upload/batch-optimized")
async def optimized_batch_upload(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    event_id: Optional[int] = None,
    process_immediately: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Optimized batch upload with progress tracking"""
    if len(files) > 50:
        raise HTTPException(status_code=413, detail="Too many files (max 50)")
    
    batch_id = hashlib.md5(f"{time.time()}{len(files)}".encode()).hexdigest()[:8]
    results = []
    upload_tasks = []
    
    # Process files concurrently
    async def process_single_file(file: UploadFile, index: int):
        try:
            contents = await file.read()
            file_hash = hashlib.sha256(contents).hexdigest()
            
            # Check duplicate
            existing = db.query(models.Image).filter_by(file_hash=file_hash).first()
            if existing:
                return {
                    "index": index,
                    "filename": file.filename,
                    "status": "duplicate",
                    "image_id": existing.id,
                    "url": existing.url
                }
            
            # Upload to CDN
            url = upload_image(contents)
            
            # Save to database
            db_image = models.Image(
                url=url,
                event_id=event_id,
                file_hash=file_hash,
                processed=0
            )
            db.add(db_image)
            db.flush()
            
            return {
                "index": index,
                "filename": file.filename,
                "status": "uploaded",
                "image_id": db_image.id,
                "url": url,
                "thumbnail": cdn_service.get_optimized_url(url, 'thumbnail')
            }
            
        except Exception as e:
            return {
                "index": index,
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            }
    
    # Process files concurrently
    tasks = [process_single_file(file, i) for i, file in enumerate(files)]
    results = await asyncio.gather(*tasks)
    
    db.commit()
    
    # Queue processing
    uploaded_images = [r for r in results if r["status"] == "uploaded"]
    if uploaded_images:
        if process_immediately:
            # Process first 10 immediately, rest in background
            immediate_batch = uploaded_images[:10]
            background_batch = uploaded_images[10:]
            
            for img in immediate_batch:
                background_tasks.add_task(
                    processing_pipeline.process_single_image,
                    img["image_id"],
                    img["url"]
                )
            
            if background_batch:
                background_tasks.add_task(
                    processing_pipeline.process_batch,
                    [(img["image_id"], img["url"]) for img in background_batch]
                )
        else:
            # All in background
            background_tasks.add_task(
                processing_pipeline.process_batch,
                [(img["image_id"], img["url"]) for img in uploaded_images]
            )
    
    # Summary statistics
    stats = {
        "uploaded": len([r for r in results if r["status"] == "uploaded"]),
        "duplicates": len([r for r in results if r["status"] == "duplicate"]),
        "errors": len([r for r in results if r["status"] == "error"])
    }
    
    return {
        "success": True,
        "batch_id": batch_id,
        "total": len(files),
        "stats": stats,
        "results": results,
        "processing_mode": "immediate" if process_immediately else "background"
    }

# Health check with detailed metrics
@app.get("/api/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Comprehensive health check"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Database health
    try:
        db.execute(text("SELECT 1"))
        db_latency_start = time.time()
        db.execute(text("SELECT COUNT(*) FROM images LIMIT 1"))
        db_latency = (time.time() - db_latency_start) * 1000
        
        health_data["services"]["database"] = {
            "status": "healthy",
            "latency_ms": round(db_latency, 2)
        }
    except Exception as e:
        health_data["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Cache health
    cache_healthy = cache_service.is_available()
    if cache_healthy:
        cache_stats = cache_service.get_cache_stats()
        health_data["services"]["cache"] = {
            "status": "healthy",
            "stats": cache_stats
        }
    else:
        health_data["services"]["cache"] = {"status": "unavailable"}
    
    # Face API health
    face_api_healthy = face_service.health_check()
    health_data["services"]["face_api"] = {
        "status": "healthy" if face_api_healthy else "unhealthy"
    }
    
    # Processing pipeline stats
    processing_stats = processing_pipeline.get_processing_stats()
    health_data["services"]["processing"] = {
        "status": "healthy",
        "stats": processing_stats
    }
    
    # Overall health
    unhealthy_services = [
        k for k, v in health_data["services"].items() 
        if v.get("status") != "healthy"
    ]
    
    if unhealthy_services:
        health_data["status"] = "degraded"
        health_data["unhealthy_services"] = unhealthy_services
    
    status_code = 200 if health_data["status"] == "healthy" else 503
    return JSONResponse(content=health_data, status_code=status_code)

# Initialize FastAPI cache
@app.on_event("startup")
async def startup_event():
    """Initialize optimizations on startup"""
    if cache_service.is_available():
        redis_backend = RedisBackend(cache_service.redis_client)
        FastAPICache.init(redis_backend, prefix="fastapi-cache")
        
        # Warm up cache
        try:
            db = next(get_db())
            cache_service.warm_up_cache(db, limit=1000)
            db.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=4,  # Multiple workers for production
        access_log=False,  # Disable for performance
        server_header=False
    )