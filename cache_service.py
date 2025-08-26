# backend/cache_service.py
import redis
import json
import numpy as np
import base64
from typing import List, Dict, Optional, Tuple
import os
from datetime import timedelta
import pickle

class CacheService:
    def __init__(self):
        # Redis connection (Railway provides REDIS_URL)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        
        try:
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=False,  # Binary data for embeddings
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            print("✓ Redis connected successfully")
        except Exception as e:
            print(f"⚠ Redis not available, using fallback: {e}")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except:
            return False
    
    # Embedding cache methods
    def cache_embedding(self, embedding_id: int, embedding: List[float], 
                       ttl_hours: int = 24) -> bool:
        """Cache single embedding"""
        if not self.redis_client:
            return False
        
        try:
            key = f"emb:{embedding_id}"
            # Serialize as binary for efficiency
            value = pickle.dumps(np.array(embedding, dtype=np.float32))
            self.redis_client.setex(
                key, 
                timedelta(hours=ttl_hours),
                value
            )
            return True
        except:
            return False
    
    def get_embedding(self, embedding_id: int) -> Optional[np.ndarray]:
        """Get cached embedding"""
        if not self.redis_client:
            return None
        
        try:
            key = f"emb:{embedding_id}"
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except:
            pass
        return None
    
    def batch_cache_embeddings(self, embeddings: Dict[int, List[float]], 
                              ttl_hours: int = 24) -> int:
        """Cache multiple embeddings at once"""
        if not self.redis_client:
            return 0
        
        try:
            pipe = self.redis_client.pipeline()
            count = 0
            
            for emb_id, embedding in embeddings.items():
                key = f"emb:{emb_id}"
                value = pickle.dumps(np.array(embedding, dtype=np.float32))
                pipe.setex(key, timedelta(hours=ttl_hours), value)
                count += 1
            
            pipe.execute()
            return count
        except:
            return 0
    
    def batch_get_embeddings(self, embedding_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get multiple cached embeddings"""
        if not self.redis_client:
            return {}
        
        try:
            pipe = self.redis_client.pipeline()
            
            for emb_id in embedding_ids:
                pipe.get(f"emb:{emb_id}")
            
            results = pipe.execute()
            embeddings = {}
            
            for emb_id, value in zip(embedding_ids, results):
                if value:
                    embeddings[emb_id] = pickle.loads(value)
            
            return embeddings
        except:
            return {}
    
    # Search result cache
    def cache_search_result(self, query_hash: str, results: List[Dict], 
                           ttl_minutes: int = 30) -> bool:
        """Cache search results"""
        if not self.redis_client:
            return False
        
        try:
            key = f"search:{query_hash}"
            value = json.dumps(results)
            self.redis_client.setex(
                key,
                timedelta(minutes=ttl_minutes),
                value
            )
            
            # Track popular searches
            self.redis_client.zincrby("popular_searches", 1, query_hash)
            
            return True
        except:
            return False
    
    def get_search_result(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached search result"""
        if not self.redis_client:
            return None
        
        try:
            key = f"search:{query_hash}"
            value = self.redis_client.get(key)
            if value:
                # Increment hit counter
                self.redis_client.hincrby("search_hits", query_hash, 1)
                return json.loads(value)
        except:
            pass
        return None
    
    # Pre-computed similarity cache
    def cache_similarity_matrix(self, image_id: int, 
                               similarities: Dict[int, float]) -> bool:
        """Cache pre-computed similarities for an image"""
        if not self.redis_client:
            return False
        
        try:
            key = f"sim:{image_id}"
            # Store as sorted set for efficient range queries
            mapping = {str(img_id): score for img_id, score in similarities.items()}
            self.redis_client.zadd(key, mapping)
            self.redis_client.expire(key, timedelta(hours=12))
            return True
        except:
            return False
    
    def get_similar_images(self, image_id: int, limit: int = 20) -> List[Tuple[int, float]]:
        """Get pre-computed similar images"""
        if not self.redis_client:
            return []
        
        try:
            key = f"sim:{image_id}"
            # Get top similar images (lowest scores)
            results = self.redis_client.zrange(key, 0, limit-1, withscores=True)
            return [(int(img_id.decode()), score) for img_id, score in results]
        except:
            return []
    
    # Statistics and monitoring
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disabled"}
        
        try:
            info = self.redis_client.info("memory")
            stats = self.redis_client.info("stats")
            
            # Get popular searches
            popular = self.redis_client.zrevrange("popular_searches", 0, 4, withscores=True)
            
            return {
                "status": "active",
                "memory_used": info.get("used_memory_human", "N/A"),
                "memory_peak": info.get("used_memory_peak_human", "N/A"),
                "total_keys": self.redis_client.dbsize(),
                "hits": stats.get("keyspace_hits", 0),
                "misses": stats.get("keyspace_misses", 0),
                "hit_rate": round(
                    stats.get("keyspace_hits", 0) / 
                    max(stats.get("keyspace_hits", 0) + stats.get("keyspace_misses", 1), 1) * 100, 
                    2
                ),
                "popular_searches": [
                    {"hash": h.decode(), "count": int(c)} 
                    for h, c in popular
                ] if popular else []
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """Clear cache by pattern or all"""
        if not self.redis_client:
            return 0
        
        try:
            if pattern:
                keys = self.redis_client.keys(f"{pattern}*")
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                self.redis_client.flushdb()
                return -1  # All cleared
        except:
            return 0
    
    # Warm up cache
    def warm_up_cache(self, db_session, limit: int = 1000):
        """Pre-load frequently accessed embeddings"""
        if not self.redis_client:
            return 0
        
        try:
            from models import FaceEmbedding, Image
            
            # Get recent high-quality faces
            recent_faces = db_session.query(
                FaceEmbedding.id,
                FaceEmbedding.embedding
            ).join(
                Image
            ).filter(
                FaceEmbedding.quality_score >= 0.5
            ).order_by(
                Image.uploaded_at.desc()
            ).limit(limit).all()
            
            # Batch cache
            embeddings_dict = {
                face.id: face.embedding 
                for face in recent_faces
            }
            
            count = self.batch_cache_embeddings(embeddings_dict, ttl_hours=48)
            print(f"✓ Warmed up cache with {count} embeddings")
            return count
            
        except Exception as e:
            print(f"Cache warm-up failed: {e}")
            return 0

# Singleton instance
cache_service = CacheService()