# backend/processing_pipeline.py
import asyncio
import logging
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
import json
from datetime import datetime

from models import Image, FaceEmbedding
from face_api_service import face_service
from cache_service import cache_service
from database import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingPipeline:
    def __init__(self, max_workers: int = 3, batch_size: int = 5):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def process_single_image(self, image_id: int, image_url: str) -> Dict:
        """Process single image with comprehensive error handling"""
        db = SessionLocal()
        start_time = time.time()
        
        try:
            # Update status to processing
            image = db.query(Image).filter_by(id=image_id).first()
            if not image:
                return {"error": "Image not found", "image_id": image_id}
            
            image.processed = 1  # Processing
            db.commit()
            
            # Extract faces with multiple attempts
            faces = self._extract_faces_with_fallback(image_url)
            
            if not faces:
                logger.warning(f"No faces detected for image {image_id}")
                image.processed = 3  # No faces
                image.face_count = 0
                db.commit()
                return {
                    "image_id": image_id,
                    "faces_count": 0,
                    "status": "no_faces",
                    "processing_time": time.time() - start_time
                }
            
            # Process and save embeddings
            embeddings_data = {}
            saved_count = 0
            
            for face in faces:
                if self._is_valid_embedding(face.get('embedding')):
                    emb_hash = hashlib.md5(
                        json.dumps(face['embedding'][:10]).encode()
                    ).hexdigest()
                    
                    embedding = FaceEmbedding(
                        image_id=image_id,
                        embedding=face['embedding'],
                        bbox=face.get('area'),
                        embedding_hash=emb_hash,
                        quality_score=max(0.1, min(1.0, face.get('confidence', 0.5)))
                    )
                    db.add(embedding)
                    db.flush()
                    
                    embeddings_data[embedding.id] = face['embedding']
                    saved_count += 1
            
            # Batch cache embeddings
            if embeddings_data and cache_service.is_available():
                cache_service.batch_cache_embeddings(embeddings_data, ttl_hours=48)
            
            # Update image status
            image.processed = 2  # Completed
            image.face_count = saved_count
            db.commit()
            
            processing_time = time.time() - start_time
            logger.info(f"Processed image {image_id}: {saved_count} faces in {processing_time:.2f}s")
            
            return {
                "image_id": image_id,
                "faces_count": saved_count,
                "status": "success",
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}")
            db.rollback()
            
            # Mark as failed
            if image:
                image.processed = -1  # Failed
            db.commit()
            
            return {
                "image_id": image_id,
                "error": str(e),
                "status": "error",
                "processing_time": time.time() - start_time
            }
        finally:
            db.close()
    
    def _extract_faces_with_fallback(self, image_url: str) -> List[Dict]:
        """Extract faces with multiple strategies"""
        # Strategy 1: All faces with high confidence
        faces = face_service.extract_faces(image_url, return_all=True)
        if faces:
            return faces
        
        # Strategy 2: Single best face with lower threshold
        faces = face_service.extract_faces(image_url, return_all=False)
        if faces:
            return faces
        
        # Strategy 3: Manual retry with delay
        time.sleep(2)
        faces = face_service.extract_faces(image_url, return_all=True)
        
        return faces
    
    def _is_valid_embedding(self, embedding) -> bool:
        """Validate embedding quality"""
        if not embedding or not isinstance(embedding, list):
            return False
        
        if len(embedding) != 128:  # FaceNet embedding size
            return False
        
        # Check for all zeros or invalid values
        if all(abs(x) < 1e-10 for x in embedding):
            return False
        
        return True
    
    def process_batch(self, image_data: List[tuple]) -> List[Dict]:
        """Process batch of images concurrently"""
        futures = []
        
        for image_id, image_url in image_data:
            future = self.executor.submit(
                self.process_single_image, 
                image_id, 
                image_url
            )
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=180)  # 3 minutes timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                results.append({"error": str(e), "status": "timeout"})
        
        return results
    
    def process_pending_images(self, limit: int = 50) -> Dict:
        """Process all pending images in batches"""
        db = SessionLocal()
        start_time = time.time()
        
        try:
            # Get pending images
            pending = db.query(
                Image.id, 
                Image.url
            ).filter(
                Image.processed == 0
            ).limit(limit).all()
            
            if not pending:
                return {
                    "status": "no_pending",
                    "processed": 0,
                    "total_time": time.time() - start_time
                }
            
            logger.info(f"Processing {len(pending)} pending images")
            
            # Process in batches
            all_results = []
            for i in range(0, len(pending), self.batch_size):
                batch = pending[i:i + self.batch_size]
                batch_results = self.process_batch(batch)
                all_results.extend(batch_results)
                
                # Small delay between batches
                if i + self.batch_size < len(pending):
                    time.sleep(1)
            
            # Summary
            success_count = sum(1 for r in all_results if r.get("status") == "success")
            error_count = sum(1 for r in all_results if r.get("status") == "error")
            no_faces_count = sum(1 for r in all_results if r.get("status") == "no_faces")
            
            total_time = time.time() - start_time
            
            logger.info(f"Batch complete: {success_count} success, {error_count} errors, {no_faces_count} no faces")
            
            return {
                "status": "completed",
                "processed": len(pending),
                "success": success_count,
                "errors": error_count,
                "no_faces": no_faces_count,
                "total_time": total_time,
                "avg_time_per_image": total_time / len(pending) if pending else 0,
                "details": all_results
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "total_time": time.time() - start_time
            }
        finally:
            db.close()
    
    def reprocess_failed_images(self, limit: int = 20) -> Dict:
        """Reprocess failed images"""
        db = SessionLocal()
        
        try:
            failed = db.query(
                Image.id, 
                Image.url
            ).filter(
                Image.processed == -1
            ).limit(limit).all()
            
            if not failed:
                return {"status": "no_failed", "processed": 0}
            
            # Reset status and reprocess
            for image_id, _ in failed:
                image = db.query(Image).filter_by(id=image_id).first()
                if image:
                    image.processed = 0
            db.commit()
            
            return self.process_pending_images(limit)
            
        except Exception as e:
            logger.error(f"Reprocess failed: {e}")
            return {"status": "error", "error": str(e)}
        finally:
            db.close()
    
    def get_processing_stats(self) -> Dict:
        """Get processing pipeline statistics"""
        db = SessionLocal()
        
        try:
            from sqlalchemy import func
            
            stats = db.query(
                func.count(Image.id).label('total'),
                func.sum(func.case([(Image.processed == 0, 1)], else_=0)).label('pending'),
                func.sum(func.case([(Image.processed == 1, 1)], else_=0)).label('processing'),
                func.sum(func.case([(Image.processed == 2, 1)], else_=0)).label('completed'),
                func.sum(func.case([(Image.processed == 3, 1)], else_=0)).label('no_faces'),
                func.sum(func.case([(Image.processed == -1, 1)], else_=0)).label('failed'),
                func.sum(Image.face_count).label('total_faces')
            ).first()
            
            return {
                "total_images": stats.total or 0,
                "pending": stats.pending or 0,
                "processing": stats.processing or 0,
                "completed": stats.completed or 0,
                "no_faces": stats.no_faces or 0,
                "failed": stats.failed or 0,
                "total_faces": stats.total_faces or 0,
                "completion_rate": round(
                    (stats.completed or 0) / max(stats.total or 1, 1) * 100, 2
                )
            }
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}
        finally:
            db.close()
    
    def cleanup_old_processing(self, hours: int = 2) -> int:
        """Cleanup images stuck in processing state"""
        db = SessionLocal()
        
        try:
            from datetime import datetime, timedelta
            cutoff = datetime.now() - timedelta(hours=hours)
            
            stuck_images = db.query(Image).filter(
                Image.processed == 1,
                Image.uploaded_at < cutoff
            ).all()
            
            count = 0
            for image in stuck_images:
                image.processed = 0  # Reset to pending
                count += 1
            
            db.commit()
            logger.info(f"Reset {count} stuck processing images")
            
            return count
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0
        finally:
            db.close()

# Singleton instance
processing_pipeline = ProcessingPipeline()


# Updated main.py background task
def process_faces_background_v2(image_id: int, url: str):
    """Optimized background processing"""
    result = processing_pipeline.process_single_image(image_id, url)
    
    if result.get("status") == "success":
        logger.info(f"✓ Background processed image {image_id}: {result['faces_count']} faces")
    else:
        logger.warning(f"✗ Background processing failed for image {image_id}: {result.get('error', 'Unknown')}")


# Add to main.py endpoints
@app.post("/api/process-batch")
async def process_batch_endpoint(
    background_tasks: BackgroundTasks,
    limit: int = Query(50, ge=1, le=100)
):
    """Process pending images in batch"""
    background_tasks.add_task(
        lambda: processing_pipeline.process_pending_images(limit)
    )
    
    return {
        "success": True,
        "message": f"Started batch processing up to {limit} images",
        "processing": "background"
    }

@app.get("/api/processing-stats")
async def get_processing_stats():
    """Get processing pipeline statistics"""
    stats = processing_pipeline.get_processing_stats()
    return {
        "success": True,
        "stats": stats
    }

@app.post("/api/cleanup-processing")
async def cleanup_processing(hours: int = Query(2, ge=1, le=24)):
    """Cleanup stuck processing images"""
    count = processing_pipeline.cleanup_old_processing(hours)
    return {
        "success": True,
        "reset_count": count,
        "message": f"Reset {count} images stuck in processing for {hours}+ hours"
    }