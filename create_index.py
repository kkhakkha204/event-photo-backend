# create_index.py
from database import get_db
from models import FaceEmbedding, EmbeddingIndex
import numpy as np

db = next(get_db())

# Index existing embeddings
embeddings = db.query(FaceEmbedding).all()

for emb in embeddings:
    if not db.query(EmbeddingIndex).filter_by(face_embedding_id=emb.id).first():
        norm = float(np.linalg.norm(emb.embedding))
        
        index = EmbeddingIndex(
            face_embedding_id=emb.id,
            norm=norm
        )
        db.add(index)

db.commit()
print(f"Indexed {len(embeddings)} embeddings")