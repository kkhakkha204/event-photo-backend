# backend/migrate_db.py
"""
Script để migrate database với schema mới
Run: python migrate_db.py
"""

from sqlalchemy import text
from database import engine
import models

def add_columns_if_not_exists():
    """Add new columns to existing tables"""
    
    with engine.connect() as conn:
        # Check and add columns to images table
        try:
            # Add file_hash column
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64) UNIQUE
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_images_file_hash 
                ON images(file_hash)
            """))
            print("✓ Added file_hash column")
        except Exception as e:
            print(f"file_hash column may already exist: {e}")
        
        try:
            # Add face_count column
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS face_count INTEGER DEFAULT 0
            """))
            print("✓ Added face_count column")
        except Exception as e:
            print(f"face_count column may already exist: {e}")
        
        try:
            # Add processed column
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS processed INTEGER DEFAULT 0
            """))
            print("✓ Added processed column")
        except Exception as e:
            print(f"processed column may already exist: {e}")
        
        # Add columns to face_embeddings table
        try:
            conn.execute(text("""
                ALTER TABLE face_embeddings 
                ADD COLUMN IF NOT EXISTS embedding_hash VARCHAR(32)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_face_embeddings_hash 
                ON face_embeddings(embedding_hash)
            """))
            print("✓ Added embedding_hash column")
        except Exception as e:
            print(f"embedding_hash column may already exist: {e}")
        
        try:
            conn.execute(text("""
                ALTER TABLE face_embeddings 
                ADD COLUMN IF NOT EXISTS quality_score FLOAT DEFAULT 0.5
            """))
            print("✓ Added quality_score column")
        except Exception as e:
            print(f"quality_score column may already exist: {e}")
        
        # Create indexes
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_images_event_id 
                ON images(event_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_images_uploaded_at 
                ON images(uploaded_at)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_face_embeddings_image_id 
                ON face_embeddings(image_id)
            """))
            print("✓ Created indexes")
        except Exception as e:
            print(f"Some indexes may already exist: {e}")
        
        # Create composite indexes
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_event_uploaded 
                ON images(event_id, uploaded_at)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_processed_uploaded 
                ON images(processed, uploaded_at)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_image_quality 
                ON face_embeddings(image_id, quality_score)
            """))
            print("✓ Created composite indexes")
        except Exception as e:
            print(f"Some composite indexes may already exist: {e}")
        
        conn.commit()

def create_new_tables():
    """Create new tables if they don't exist"""
    
    # Create all tables from models
    models.Base.metadata.create_all(bind=engine)
    print("✓ Created new tables (SearchCache, EmbeddingIndex)")

def update_existing_data():
    """Update existing records with default values"""
    
    with engine.connect() as conn:
        # Set processed = 2 for all existing images (assume they're done)
        conn.execute(text("""
            UPDATE images 
            SET processed = 2 
            WHERE processed IS NULL OR processed = 0
        """))
        
        # Count faces for existing images
        conn.execute(text("""
            UPDATE images 
            SET face_count = (
                SELECT COUNT(*) 
                FROM face_embeddings 
                WHERE face_embeddings.image_id = images.id
            )
            WHERE face_count IS NULL OR face_count = 0
        """))
        
        # Set default quality score
        conn.execute(text("""
            UPDATE face_embeddings 
            SET quality_score = 0.5 
            WHERE quality_score IS NULL
        """))
        
        conn.commit()
        print("✓ Updated existing data with defaults")

def cleanup_duplicates():
    """Remove duplicate faces if any"""
    
    with engine.connect() as conn:
        # Find and remove duplicate embeddings for same image
        result = conn.execute(text("""
            DELETE FROM face_embeddings 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM face_embeddings 
                GROUP BY image_id, embedding_hash
            )
        """))
        
        if result.rowcount > 0:
            print(f"✓ Removed {result.rowcount} duplicate embeddings")
        else:
            print("✓ No duplicate embeddings found")
        
        conn.commit()

def vacuum_analyze():
    """Optimize database after changes"""
    
    # VACUUM cần chạy ngoài transaction, sử dụng autocommit
    with engine.connect() as conn:
        conn.execute(text("COMMIT"))  # End any existing transaction
        
        # Set autocommit mode
        conn = conn.execution_options(autocommit=True)
        
        try:
            conn.execute(text("VACUUM ANALYZE images"))
            conn.execute(text("VACUUM ANALYZE face_embeddings"))
            print("✓ Database optimized")
        except Exception as e:
            print(f"Warning: Could not optimize database: {e}")
            print("This is not critical for functionality")

if __name__ == "__main__":
    print("Starting database migration...")
    
    try:
        # Step 1: Add new columns
        print("\n1. Adding new columns...")
        add_columns_if_not_exists()
        
        # Step 2: Create new tables
        print("\n2. Creating new tables...")
        create_new_tables()
        
        # Step 3: Update existing data
        print("\n3. Updating existing data...")
        update_existing_data()
        
        # Step 4: Clean duplicates
        print("\n4. Cleaning duplicates...")
        cleanup_duplicates()
        
        # Step 5: Optimize database
        print("\n5. Optimizing database...")
        vacuum_analyze()
        
        print("\n✅ Migration completed successfully!")
        
        # Print statistics
        with engine.connect() as conn:
            images = conn.execute(text("SELECT COUNT(*) FROM images")).scalar()
            faces = conn.execute(text("SELECT COUNT(*) FROM face_embeddings")).scalar()
            print(f"\nDatabase stats:")
            print(f"- Total images: {images}")
            print(f"- Total faces: {faces}")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Please fix the error and run again")