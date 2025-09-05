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
    Nén ảnh để giảm kích thước file
    max_size_mb: kích thước tối đa (MB)
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Nếu file đã nhỏ hơn giới hạn thì return nguyên bản
    if len(file_content) <= max_size_bytes:
        return file_content
    
    try:
        # Mở ảnh
        image = Image.open(io.BytesIO(file_content))
        
        # Chuyển sang RGB nếu cần
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        
        # Thử giảm chất lượng trước
        for quality in [85, 75, 65, 55, 45]:
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=quality, optimize=True)
            compressed_content = output.getvalue()
            
            if len(compressed_content) <= max_size_bytes:
                print(f"✓ Compressed image: {len(file_content)/1024/1024:.1f}MB → {len(compressed_content)/1024/1024:.1f}MB (quality={quality})")
                return compressed_content
        
        # Nếu vẫn quá lớn, resize ảnh
        original_width, original_height = image.size
        
        for scale in [0.9, 0.8, 0.7, 0.6, 0.5]:
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            resized_image = image.resize((new_width, new_height), Image.Lanczos)
            
            # Thử các quality khác nhau
            for quality in [75, 65, 55]:
                output = io.BytesIO()
                resized_image.save(output, format='JPEG', quality=quality, optimize=True)
                compressed_content = output.getvalue()
                
                if len(compressed_content) <= max_size_bytes:
                    print(f"✓ Resized and compressed: {original_width}x{original_height} → {new_width}x{new_height}, {len(file_content)/1024/1024:.1f}MB → {len(compressed_content)/1024/1024:.1f}MB")
                    return compressed_content
        
        # Nếu vẫn không được, trả về phiên bản nén tối đa
        output = io.BytesIO()
        final_image = image.resize((int(original_width * 0.4), int(original_height * 0.4)), Image.Lanczos)
        final_image.save(output, format='JPEG', quality=40, optimize=True)
        return output.getvalue()
        
    except Exception as e:
        print(f"✗ Error compressing image: {e}")
        # Nếu có lỗi, trả về nguyên bản
        return file_content

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