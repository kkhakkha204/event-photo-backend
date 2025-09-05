import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from PIL import Image
import io

load_dotenv()

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def compress_image(file_content: bytes, max_size_mb: float = 9.0) -> bytes:
    """
    Nén ảnh thông minh để bảo tồn face detection accuracy
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if len(file_content) <= max_size_bytes:
        return file_content
    
    try:
        image = Image.open(io.BytesIO(file_content))
        
        # Chuyển sang RGB nếu cần
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        original_width, original_height = image.size
        print(f"Original size: {original_width}x{original_height}, {len(file_content)/1024/1024:.1f}MB")
        
        # STRATEGY 1: Giữ nguyên kích thước, chỉ giảm quality nhẹ
        # Face detection hoạt động tốt nhất với resolution cao
        for quality in [90, 85, 80, 75]:
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_content = output.getvalue()
            
            if len(compressed_content) <= max_size_bytes:
                print(f"✓ Quality compression: {len(file_content)/1024/1024:.1f}MB → {len(compressed_content)/1024/1024:.1f}MB (Q={quality})")
                return compressed_content
        
        # STRATEGY 2: Resize thông minh - giữ tỷ lệ faces
        # Chỉ resize khi thực sự cần thiết
        if original_width > 2000 or original_height > 2000:
            # Resize xuống mức vừa phải để giữ chi tiết faces
            max_dimension = 1600  # Đủ lớn cho face detection
            if original_width > original_height:
                new_width = max_dimension
                new_height = int(original_height * max_dimension / original_width)
            else:
                new_height = max_dimension
                new_width = int(original_width * max_dimension / original_height)
            
            # Sử dụng Lanczos resampling để giữ chi tiết tốt nhất
            resized_image = image.resize((new_width, new_height), Image.Lanczos)
            
            for quality in [85, 80, 75, 70]:
                output = io.BytesIO()
                resized_image.save(output, format='JPEG', quality=quality, optimize=True)
                compressed_content = output.getvalue()
                
                if len(compressed_content) <= max_size_bytes:
                    print(f"✓ Smart resize: {original_width}x{original_height} → {new_width}x{new_height}, Q={quality}")
                    return compressed_content
        
        # STRATEGY 3: Progressive resize với quality cao
        for scale in [0.85, 0.75, 0.65]:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Đảm bảo không resize quá nhỏ (tối thiểu 800px cho cạnh lớn nhất)
            max_side = max(new_width, new_height)
            if max_side < 800:
                continue
            
            resized_image = image.resize((new_width, new_height), Image.Lanczos)
            
            # Dùng quality cao hơn cho ảnh đã resize
            for quality in [80, 75, 70]:
                output = io.BytesIO()
                resized_image.save(output, format='JPEG', quality=quality, optimize=True)
                compressed_content = output.getvalue()
                
                if len(compressed_content) <= max_size_bytes:
                    print(f"✓ Progressive resize: {new_width}x{new_height}, Q={quality}")
                    return compressed_content
        
        # FALLBACK: Nếu vẫn không được, báo lỗi thay vì nén quá mạnh
        print(f"✗ Cannot compress without losing too much quality")
        raise Exception("Image too large and cannot be compressed safely for face detection")
        
    except Exception as e:
        print(f"Compression error: {e}")
        # Thay vì return nguyên bản, raise error để user biết
        raise e

def upload_image(file):
    """Upload image to Cloudinary với nén ảnh tự động"""
    
    # Nếu file là bytes, nén trước khi upload
    if isinstance(file, bytes):
        file = compress_image(file)
    
    try:
        result = cloudinary.uploader.upload(
            file,
            # Tối ưu thêm từ Cloudinary
            transformation=[
                {'quality': 'auto:good'},  # Auto optimize quality
                {'fetch_format': 'auto'}   # Auto choose best format
            ]
        )
        return result['secure_url']
        
    except cloudinary.exceptions.Error as e:
        error_msg = str(e)
        
        # Nếu vẫn quá lớn, thử nén mạnh hơn
        if "File size too large" in error_msg and isinstance(file, bytes):
            print("⚠️ File still too large, trying maximum compression...")
            file = compress_image(file, max_size_mb=5.0)  # Giảm xuống 5MB
            
            result = cloudinary.uploader.upload(
                file,
                transformation=[
                    {'quality': 'auto:low'},
                    {'fetch_format': 'auto'}
                ]
            )
            return result['secure_url']
        else:
            raise e