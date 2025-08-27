from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, Index, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
import hashlib
import json

class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    event_id = Column(Integer, nullable=True, index=True)  # Index for filtering
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)  # Index for sorting
    
    # Thêm fields để optimize
    file_hash = Column(String(64), unique=True, index=True)  # Detect duplicates
    face_count = Column(Integer, default=0)  # Cache số faces
    processed = Column(Integer, default=0)  # 0=pending, 1=processing, 2=done
    
    # Relationships
    face_embeddings = relationship("FaceEmbedding", back_populates="image", cascade="all, delete-orphan")
    
    # Composite indexes cho queries phức tạp
    __table_args__ = (
        Index('idx_event_uploaded', 'event_id', 'uploaded_at'),
        Index('idx_processed_uploaded', 'processed', 'uploaded_at'),
    )

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), index=True)  # Index for JOIN
    embedding = Column(JSON, nullable=False)  # Keep JSON for compatibility
    bbox = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Thêm fields để optimize search
    embedding_hash = Column(String(32), index=True)  # Quick lookup
    quality_score = Column(Float, default=0.0)  # Face quality/confidence
    
    # Relationship
    image = relationship("Image", back_populates="face_embeddings")
    
    # Composite index cho image_id + quality
    __table_args__ = (
        Index('idx_image_quality', 'image_id', 'quality_score'),
    )

# Thêm bảng cache cho frequent searches
class SearchCache(Base):
    __tablename__ = "search_cache"
    
    id = Column(Integer, primary_key=True)
    query_hash = Column(String(64), unique=True, index=True)
    results = Column(JSON)  # Cached results
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    hit_count = Column(Integer, default=1)
    
    # Auto cleanup old cache
    __table_args__ = (
        Index('idx_cache_created', 'created_at'),
    )

# Bảng pre-computed embeddings cho fast search (binary format)
class EmbeddingIndex(Base):
    __tablename__ = "embedding_index"
    
    id = Column(Integer, primary_key=True)
    face_embedding_id = Column(Integer, ForeignKey("face_embeddings.id"), unique=True)
    
    # Store embedding as 128 float columns for fast SQL operations
    # This allows us to use SQL for distance calculations
    embedding_binary = Column(Text)  # Base64 encoded numpy array
    
    # Pre-computed norms cho cosine similarity
    norm = Column(Float, index=True)
    
    __table_args__ = (
        Index('idx_embedding_norm', 'norm'),
    )