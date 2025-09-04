# debug_faces.py
import requests
from sqlalchemy.orm import Session
from database import get_db
from models import Image, FaceEmbedding
import json

def check_detection_status():
    """Kiểm tra tỷ lệ detect faces"""
    db = next(get_db())
    
    # Tổng số ảnh
    total_images = db.query(Image).count()
    processed_images = db.query(Image).filter_by(processed=2).count()
    unprocessed_images = db.query(Image).filter_by(processed=0).count()
    
    # Ảnh có faces vs không có faces
    images_with_faces = db.query(Image).filter(Image.face_count > 0).count()
    images_no_faces = db.query(Image).filter(Image.face_count == 0).count()
    
    # Quality distribution
    low_quality = db.query(FaceEmbedding).filter(FaceEmbedding.quality_score < 0.3).count()
    medium_quality = db.query(FaceEmbedding).filter(
        FaceEmbedding.quality_score.between(0.3, 0.6)
    ).count()
    high_quality = db.query(FaceEmbedding).filter(FaceEmbedding.quality_score > 0.6).count()
    
    print("=== DETECTION STATUS ===")
    print(f"Total images: {total_images}")
    print(f"Processed: {processed_images} ({processed_images/total_images*100:.1f}%)")
    print(f"Unprocessed: {unprocessed_images}")
    print(f"Images with faces: {images_with_faces} ({images_with_faces/processed_images*100:.1f}%)")
    print(f"Images no faces: {images_no_faces} ({images_no_faces/processed_images*100:.1f}%)")
    print(f"\nQuality distribution:")
    print(f"Low (<0.3): {low_quality}")
    print(f"Medium (0.3-0.6): {medium_quality}")  
    print(f"High (>0.6): {high_quality}")
    
    # Sample images without faces
    no_face_images = db.query(Image).filter(Image.face_count == 0).limit(5).all()
    print(f"\nSample images without faces:")
    for img in no_face_images:
        print(f"ID {img.id}: {img.url}")
    
    db.close()

def test_face_api_directly():
    """Test Face API trực tiếp"""
    db = next(get_db())
    
    # Lấy 10 ảnh không có faces
    no_face_images = db.query(Image).filter(Image.face_count == 0).limit(10).all()
    
    print("=== TESTING FACE API DIRECTLY ===")
    for img in no_face_images:
        try:
            response = requests.post('http://localhost:5000/detect', json={
                'image_url': img.url,
                'return_all_faces': True
            }, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                faces = data.get('faces', [])
                print(f"Image {img.id}: Found {len(faces)} faces")
                
                if len(faces) > 0:
                    print(f"  -> Should have {len(faces)} faces but DB shows 0!")
                    for i, face in enumerate(faces):
                        print(f"     Face {i}: confidence={face.get('confidence', 0):.3f}")
            else:
                print(f"Image {img.id}: API Error {response.status_code}")
                
        except Exception as e:
            print(f"Image {img.id}: Exception - {e}")
    
    db.close()

def test_search_accuracy():
    """Test độ chính xác search"""
    db = next(get_db())
    
    # Lấy 1 ảnh có face quality cao
    test_image = db.query(Image).join(FaceEmbedding).filter(
        FaceEmbedding.quality_score > 0.7
    ).first()
    
    if not test_image:
        print("No high quality face found for testing")
        return
    
    print(f"=== TESTING SEARCH ACCURACY ===")
    print(f"Test image: {test_image.id} - {test_image.url}")
    
    # Download và search với chính ảnh đó
    try:
        import requests
        response = requests.get(test_image.url)
        
        files = {'file': ('test.jpg', response.content, 'image/jpeg')}
        search_response = requests.post(
            'http://localhost:8000/api/search?mode=loose&limit=50',
            files=files
        )
        
        if search_response.status_code == 200:
            results = search_response.json()
            if results.get('success'):
                found_results = results.get('results', [])
                print(f"Search returned {len(found_results)} results")
                
                # Kiểm tra xem có tìm thấy chính ảnh đó không
                found_self = any(r.get('image_id') == test_image.id for r in found_results)
                print(f"Found self: {found_self}")
                
                if found_results:
                    best_match = found_results[0]
                    print(f"Best match: ID {best_match.get('image_id')}, distance: {best_match.get('min_distance'):.3f}")
                
                # Check thresholds used
                thresholds = results.get('thresholds', {})
                print(f"Thresholds: {thresholds}")
            else:
                print(f"Search failed: {results.get('message')}")
        else:
            print(f"Search API error: {search_response.status_code}")
            
    except Exception as e:
        print(f"Search test failed: {e}")
    
    db.close()

def reprocess_failed_images():
    """Reprocess ảnh không có faces"""
    db = next(get_db())
    
    failed_images = db.query(Image).filter(Image.face_count == 0).limit(20).all()
    
    print(f"=== REPROCESSING {len(failed_images)} FAILED IMAGES ===")
    
    for img in failed_images:
        try:
            # Call face API
            response = requests.post('http://localhost:5000/detect', json={
                'image_url': img.url,
                'return_all_faces': True
            }, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                faces = data.get('faces', [])
                
                if len(faces) > 0:
                    print(f"Image {img.id}: Found {len(faces)} faces - updating DB")
                    
                    # Delete old embeddings
                    db.query(FaceEmbedding).filter_by(image_id=img.id).delete()
                    
                    # Add new embeddings
                    for face in faces:
                        embedding = FaceEmbedding(
                            image_id=img.id,
                            embedding=face['embedding'],
                            bbox=face.get('area'),
                            quality_score=face.get('confidence', 0.5)
                        )
                        db.add(embedding)
                    
                    # Update image
                    img.face_count = len(faces)
                    img.processed = 2
                    
                    db.commit()
                else:
                    print(f"Image {img.id}: Still no faces found")
            else:
                print(f"Image {img.id}: API error {response.status_code}")
                
        except Exception as e:
            print(f"Image {img.id}: Error - {e}")
            db.rollback()
    
    db.close()

if __name__ == "__main__":
    print("Face Detection Debug Tool")
    print("1. Check detection status")
    print("2. Test Face API directly")  
    print("3. Test search accuracy")
    print("4. Reprocess failed images")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "1":
        check_detection_status()
    elif choice == "2":
        test_face_api_directly()
    elif choice == "3":
        test_search_accuracy()
    elif choice == "4":
        reprocess_failed_images()
    else:
        print("Invalid choice")