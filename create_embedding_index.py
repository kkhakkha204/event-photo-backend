# create_embedding_index.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from database import get_db, DATABASE_URL, engine
from models import FaceEmbedding, EmbeddingIndex, Image
import numpy as np
import base64
from tqdm import tqdm
import hashlib

def create_index_table():
    """Create EmbeddingIndex table if not exists"""
    try:
        # Create table using raw SQL to ensure it exists
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS embedding_index (
                    id SERIAL PRIMARY KEY,
                    face_embedding_id INTEGER UNIQUE REFERENCES face_embeddings(id) ON DELETE CASCADE,
                    embedding_binary TEXT,
                    norm FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_embedding_norm ON embedding_index(norm);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_embedding_face_id ON embedding_index(face_embedding_id);
            """))
            
            conn.commit()
            print("✓ EmbeddingIndex table created successfully")
            
    except Exception as e:
        print(f"Error creating table: {e}")
        return False
    
    return True

def populate_embedding_index():
    """Populate EmbeddingIndex with existing face embeddings"""
    db = next(get_db())
    
    try:
        # Check existing index count
        existing_count = db.query(EmbeddingIndex).count()
        print(f"Existing index entries: {existing_count}")
        
        # Get all face embeddings that don't have index yet
        embeddings = db.query(FaceEmbedding).outerjoin(
            EmbeddingIndex,
            FaceEmbedding.id == EmbeddingIndex.face_embedding_id
        ).filter(
            EmbeddingIndex.face_embedding_id == None
        ).all()
        
        print(f"Found {len(embeddings)} embeddings to index")
        
        if len(embeddings) == 0:
            print("All embeddings already indexed!")
            return True
        
        # Process in batches
        batch_size = 100
        total_processed = 0
        
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Creating indexes"):
            batch = embeddings[i:i + batch_size]
            
            for emb in batch:
                try:
                    # Calculate norm
                    embedding_array = np.array(emb.embedding, dtype=np.float32)
                    norm = float(np.linalg.norm(embedding_array))
                    
                    # Create binary representation for future use
                    embedding_binary = base64.b64encode(embedding_array.tobytes()).decode()
                    
                    # Create index entry
                    index_entry = EmbeddingIndex(
                        face_embedding_id=emb.id,
                        norm=norm,
                        embedding_binary=embedding_binary
                    )
                    
                    db.add(index_entry)
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing embedding {emb.id}: {e}")
                    continue
            
            # Commit batch
            try:
                db.commit()
                print(f"Processed batch {i//batch_size + 1}/{(len(embeddings) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error committing batch: {e}")
                db.rollback()
        
        print(f"✓ Successfully indexed {total_processed} embeddings")
        return True
        
    except Exception as e:
        print(f"Error populating index: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def verify_index():
    """Verify the index was created correctly"""
    db = next(get_db())
    
    try:
        # Count statistics
        total_embeddings = db.query(FaceEmbedding).count()
        indexed_embeddings = db.query(EmbeddingIndex).count()
        
        print(f"\n=== INDEX VERIFICATION ===")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Indexed embeddings: {indexed_embeddings}")
        print(f"Coverage: {indexed_embeddings/total_embeddings*100:.1f}%")
        
        # Sample norm distribution
        norms = db.query(EmbeddingIndex.norm).limit(1000).all()
        if norms:
            norm_values = [n.norm for n in norms]
            print(f"Norm range: {min(norm_values):.3f} - {max(norm_values):.3f}")
            print(f"Norm average: {np.mean(norm_values):.3f}")
        
        # Test query performance
        import time
        start_time = time.time()
        
        test_results = db.query(EmbeddingIndex).filter(
            EmbeddingIndex.norm.between(10.0, 12.0)
        ).limit(100).all()
        
        end_time = time.time()
        print(f"Query test: {len(test_results)} results in {end_time - start_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"Error verifying index: {e}")
        return False
    finally:
        db.close()

def clean_orphaned_indexes():
    """Clean up any orphaned index entries"""
    db = next(get_db())
    
    try:
        # Find orphaned indexes (where face_embedding was deleted)
        orphaned = db.query(EmbeddingIndex).outerjoin(
            FaceEmbedding,
            EmbeddingIndex.face_embedding_id == FaceEmbedding.id
        ).filter(
            FaceEmbedding.id == None
        ).all()
        
        if orphaned:
            print(f"Found {len(orphaned)} orphaned indexes, cleaning up...")
            for index in orphaned:
                db.delete(index)
            db.commit()
            print("✓ Cleaned up orphaned indexes")
        else:
            print("No orphaned indexes found")
            
    except Exception as e:
        print(f"Error cleaning orphaned indexes: {e}")
        db.rollback()
    finally:
        db.close()

def rebuild_index():
    """Rebuild the entire index from scratch"""
    db = next(get_db())
    
    try:
        print("Rebuilding entire embedding index...")
        
        # Clear existing index
        db.query(EmbeddingIndex).delete()
        db.commit()
        print("✓ Cleared existing index")
        
        # Repopulate
        result = populate_embedding_index()
        
        if result:
            print("✓ Index rebuild completed successfully")
        else:
            print("✗ Index rebuild failed")
            
        return result
        
    except Exception as e:
        print(f"Error rebuilding index: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("EmbeddingIndex Management Tool")
    print("1. Create index table")
    print("2. Populate index")
    print("3. Verify index")
    print("4. Clean orphaned indexes")
    print("5. Rebuild entire index")
    print("6. Full setup (create + populate + verify)")
    
    choice = input("Choose option (1-6): ").strip()
    
    if choice == "1":
        create_index_table()
    elif choice == "2":
        populate_embedding_index()
    elif choice == "3":
        verify_index()
    elif choice == "4":
        clean_orphaned_indexes()
    elif choice == "5":
        rebuild_index()
    elif choice == "6":
        # Full setup
        print("Running full setup...")
        if create_index_table():
            if populate_embedding_index():
                verify_index()
                clean_orphaned_indexes()
                print("\n✓ Full setup completed successfully!")
            else:
                print("\n✗ Setup failed during population")
        else:
            print("\n✗ Setup failed during table creation")
    else:
        print("Invalid choice")