from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from database import Base

class Image(Base):
    tablename = "images"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    event_id = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

class FaceEmbedding(Base):
    tablename = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    embedding = Column(JSON, nullable=False)
    bbox = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())