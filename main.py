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
        
        # Extract face with additional metadata
        query_faces = face_service.extract_faces(query_url, return_all=False)
        
        if not query_faces:
            return {
                "success": False,
                "message": "No face found in query image",
                "query_url": query_url
            }
        
        query_face = query_faces[0]
        query_embedding = np.array(query_face['embedding'])
        
        # Calculate adaptive thresholds based on query face characteristics
        adaptive_thresholds = get_adaptive_threshold(query_faces, mode)
        distance_threshold = adaptive_thresholds['distance_threshold']
        min_quality = adaptive_thresholds['min_quality']
        
        # Generate cache key including face characteristics
        query_hash = calculate_enhanced_query_hash(
            query_embedding.tolist(), 
            mode, 
            query_face.get('area', {}),
            query_face.get('quality_score', 0)
        )
        
        query_norm = float(np.linalg.norm(query_embedding))
        
        # Check Redis cache first (commented out in original, keeping structure)
        # if cache_service.is_available():
        #     cached_result = cache_service.get_search_result(query_hash)
        #     if cached_result:
        #         # Process cached results...
        #         return cached_response
        
        # Enhanced database query with adaptive quality filtering
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
                # Enhanced norm-based pre-filtering
                models.EmbeddingIndex.norm.between(
                    query_norm - adaptive_thresholds['norm_range'], 
                    query_norm + adaptive_thresholds['norm_range']
                )
            )
        )
        
        # Adaptive result set size based on query face quality
        max_candidates = adaptive_thresholds['max_candidates']
        if mode == "strict":
            query = query.order_by(desc(models.Image.uploaded_at)).limit(max_candidates)
        else:
            query = query.limit(max_candidates + 2000)
        
        results = query.all()
        
        # Prepare embeddings data
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
        
        # Enhanced vectorized search with adaptive threshold
        matches = await vectorized_search_adaptive(
            query_embedding, 
            embeddings_data, 
            distance_threshold,
            query_face
        )
        
        # Group by image with enhanced scoring
        image_scores = {}
        for match in matches:
            img_id = match['image_id']
            if img_id not in image_scores:
                image_scores[img_id] = {
                    'url': match['url'],
                    'distances': [],
                    'bboxes': [],
                    'qualities': [],
                    'best_quality': 0,
                    'uploaded_at': match['uploaded_at']
                }
            
            image_scores[img_id]['distances'].append(match['distance'])
            image_scores[img_id]['bboxes'].append(match['bbox'])
            image_scores[img_id]['qualities'].append(match['quality'])
            image_scores[img_id]['best_quality'] = max(
                image_scores[img_id]['best_quality'], 
                match['quality']
            )
        
        # Build final results with enhanced scoring
        final_results = []
        for img_id, data in image_scores.items():
            min_distance = min(data['distances'])
            avg_distance = np.mean(data['distances'])
            quality_boost = calculate_quality_boost(data['qualities'], query_face.get('quality_score', 0.5))
            
            # Enhanced composite score with adaptive weights
            weights = adaptive_thresholds['scoring_weights']
            composite_score = (
                min_distance * weights['min_distance'] + 
                avg_distance * weights['avg_distance'] + 
                (1 / (len(data['distances']) + 1)) * weights['face_count'] +
                (1 - data['best_quality']) * weights['quality'] -
                quality_boost * weights['quality_boost']  # Subtract to favor better quality matches
            )
            
            # Enhanced confidence calculation
            confidence = calculate_adaptive_confidence(
                min_distance, 
                adaptive_thresholds['confidence_params'],
                query_face.get('quality_score', 0.5)
            )
            
            result = {
                'image_id': img_id,
                'url': data['url'],
                'optimized_urls': cdn_service.get_responsive_urls(data['url']),
                'min_distance': float(min_distance),
                'avg_distance': float(avg_distance),
                'face_count': len(data['distances']),
                'composite_score': float(composite_score),
                'confidence': float(confidence),
                'quality_boost': float(quality_boost),
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
            "query_face_info": {
                "area": query_face.get('area', {}),
                "quality_score": query_face.get('quality_score', 0),
                "adaptive_mode": adaptive_thresholds['mode_info']
            },
            "results": final_results,
            "total_matches": len(final_results),
            "adaptive_thresholds": {
                "distance_threshold": distance_threshold,
                "min_quality": min_quality,
                "mode": mode,
                "query_face_category": adaptive_thresholds['face_category']
            },
            "mode": mode,
            "from_cache": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for adaptive threshold system

def get_adaptive_threshold(query_faces, mode):
    """
    Calculate adaptive thresholds based on query face characteristics
    """
    face = query_faces[0]
    face_area = face.get('area', {})
    
    # Calculate face area (width * height)
    if isinstance(face_area, dict) and 'w' in face_area and 'h' in face_area:
        area_pixels = face_area['w'] * face_area['h']
    else:
        # Fallback if area not available
        area_pixels = 15000  # Assume medium size
    
    # Get face quality score
    face_quality = face.get('quality_score', 0.5)
    
    # Categorize face size
    if area_pixels < 8000:  # Very small face
        face_category = "very_small"
        base_thresholds = {
            'strict': {'distance': 0.50, 'quality': 0.15},
            'balanced': {'distance': 0.60, 'quality': 0.20}, 
            'loose': {'distance': 0.70, 'quality': 0.25}
        }
        norm_range = 1.0
        max_candidates = 8000
    elif area_pixels < 15000:  # Small face
        face_category = "small"
        base_thresholds = {
            'strict': {'distance': 0.45, 'quality': 0.20},
            'balanced': {'distance': 0.55, 'quality': 0.25}, 
            'loose': {'distance': 0.65, 'quality': 0.30}
        }
        norm_range = 0.9
        max_candidates = 6000
    elif area_pixels < 30000:  # Medium face
        face_category = "medium"
        base_thresholds = {
            'strict': {'distance': 0.35, 'quality': 0.25},
            'balanced': {'distance': 0.45, 'quality': 0.30}, 
            'loose': {'distance': 0.55, 'quality': 0.35}
        }
        norm_range = 0.8
        max_candidates = 4000
    else:  # Large face
        face_category = "large"
        base_thresholds = {
            'strict': {'distance': 0.30, 'quality': 0.30},
            'balanced': {'distance': 0.40, 'quality': 0.35}, 
            'loose': {'distance': 0.50, 'quality': 0.40}
        }
        norm_range = 0.7
        max_candidates = 4000
    
    # Adjust thresholds based on query face quality
    quality_factor = 1.0
    if face_quality < 0.3:  # Low quality query
        quality_factor = 1.15  # Relax thresholds
    elif face_quality > 0.7:  # High quality query
        quality_factor = 0.9   # Tighten thresholds
    
    selected_thresholds = base_thresholds[mode]
    
    return {
        'distance_threshold': selected_thresholds['distance'] * quality_factor,
        'min_quality': selected_thresholds['quality'],
        'norm_range': norm_range,
        'max_candidates': max_candidates,
        'face_category': face_category,
        'scoring_weights': get_scoring_weights(face_category, mode),
        'confidence_params': get_confidence_params(face_category),
        'mode_info': {
            'area_pixels': area_pixels,
            'quality_factor': quality_factor,
            'base_mode': mode
        }
    }


def get_scoring_weights(face_category, mode):
    """
    Get adaptive scoring weights based on face category and mode
    """
    base_weights = {
        'min_distance': 0.4,
        'avg_distance': 0.3,
        'face_count': 0.2,
        'quality': 0.1,
        'quality_boost': 0.05
    }
    
    # Adjust weights for small faces (prioritize quality more)
    if face_category in ['very_small', 'small']:
        base_weights['quality'] += 0.05
        base_weights['quality_boost'] += 0.03
        base_weights['min_distance'] -= 0.05
        base_weights['avg_distance'] -= 0.03
    
    # Adjust for strict mode (prioritize distance more)
    if mode == 'strict':
        base_weights['min_distance'] += 0.1
        base_weights['face_count'] -= 0.05
        base_weights['quality'] -= 0.05
    
    return base_weights


def get_confidence_params(face_category):
    """
    Get confidence calculation parameters based on face category
    """
    if face_category == 'very_small':
        return {'base_confidence': 0.6, 'distance_penalty': 1.5}
    elif face_category == 'small':
        return {'base_confidence': 0.7, 'distance_penalty': 1.3}
    elif face_category == 'medium':
        return {'base_confidence': 0.8, 'distance_penalty': 1.0}
    else:  # large
        return {'base_confidence': 0.85, 'distance_penalty': 0.9}


def calculate_enhanced_query_hash(embedding, mode, face_area, quality_score):
    """
    Enhanced hash calculation including face characteristics
    """
    area_key = f"{face_area.get('w', 0)}x{face_area.get('h', 0)}" if isinstance(face_area, dict) else "unknown"
    quality_key = f"q{int(quality_score * 100)}" if quality_score else "q50"
    
    embedding_hash = hashlib.md5(str(embedding).encode()).hexdigest()[:16]
    return f"{embedding_hash}_{mode}_{area_key}_{quality_key}"


async def vectorized_search_adaptive(query_embedding, embeddings_data, threshold, query_face):
    """
    Enhanced vectorized search with adaptive filtering
    """
    if not embeddings_data:
        return []
    
    # Convert to numpy arrays
    embeddings = np.array([item['embedding'] for item in embeddings_data])
    
    # Calculate distances
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    
    # Find matches within threshold
    valid_indices = np.where(distances <= threshold)[0]
    
    matches = []
    for idx in valid_indices:
        item = embeddings_data[idx]
        matches.append({
            'image_id': item['image_id'],
            'url': item['url'],
            'distance': float(distances[idx]),
            'bbox': item['bbox'],
            'quality': item['quality'],
            'uploaded_at': item['uploaded_at']
        })
    
    return matches


def calculate_quality_boost(match_qualities, query_quality):
    """
    Calculate quality boost factor for scoring
    """
    if not match_qualities:
        return 0.0
    
    avg_match_quality = np.mean(match_qualities)
    max_match_quality = max(match_qualities)
    
    # Boost score if matches have significantly better quality than query
    if max_match_quality > query_quality + 0.1:
        return min(0.1, (max_match_quality - query_quality) * 0.5)
    
    return 0.0


def calculate_adaptive_confidence(distance, confidence_params, query_quality):
    """
    Calculate adaptive confidence score
    """
    base_confidence = confidence_params['base_confidence']
    distance_penalty = confidence_params['distance_penalty']
    
    # Base confidence calculation
    confidence = max(0, base_confidence - (distance * distance_penalty))
    
    # Adjust based on query quality
    if query_quality > 0.6:
        confidence = min(0.95, confidence * 1.1)  # Boost for high-quality queries
    elif query_quality < 0.3:
        confidence = confidence * 0.9  # Penalize for low-quality queries
    
    return max(0.1, min(0.95, confidence))

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