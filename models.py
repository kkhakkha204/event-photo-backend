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
    event_id = Column(Integer, nullable=True, index=True)  
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)  
    
    file_hash = Column(String(64), unique=True, index=True)  
    face_count = Column(Integer, default=0) 
    processed = Column(Integer, default=0)  
    
    face_embeddings = relationship("FaceEmbedding", back_populates="image", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_event_uploaded', 'event_id', 'uploaded_at'),
        Index('idx_processed_uploaded', 'processed', 'uploaded_at'),
    )

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), index=True)  
    embedding = Column(JSON, nullable=False)  
    bbox = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    embedding_hash = Column(String(32), index=True)  
    quality_score = Column(Float, default=0.0)  
    
    image = relationship("Image", back_populates="face_embeddings")
    
    __table_args__ = (
        Index('idx_image_quality', 'image_id', 'quality_score'),
    )

class SearchCache(Base):
    __tablename__ = "search_cache"
    
    id = Column(Integer, primary_key=True)
    query_hash = Column(String(64), unique=True, index=True)
    results = Column(JSON)  
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    hit_count = Column(Integer, default=1)
    
    __table_args__ = (
        Index('idx_cache_created', 'created_at'),
    )

class EmbeddingIndex(Base):
    __tablename__ = "embedding_index"
    
    id = Column(Integer, primary_key=True)
    face_embedding_id = Column(Integer, ForeignKey("face_embeddings.id"), unique=True)
    
    embedding_binary = Column(Text)  
    
    norm = Column(Float, index=True)
    
    __table_args__ = (
        Index('idx_embedding_norm', 'norm'),
    )