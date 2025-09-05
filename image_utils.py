# backend/image_utils.py
from PIL import Image, ImageOps, ExifTags
import io
import os
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Optimized image processing for face detection while maintaining quality
    """
    
    def __init__(self):
        self.max_file_size = 9 * 1024 * 1024  # 9MB (close to 10MB limit)
        self.max_dimension = 2048  # Higher resolution for better quality
        self.min_dimension = 640   # Minimum for good face detection
        self.quality_levels = [95, 90, 85, 82, 78, 75, 72]  # Higher quality levels
    
    def get_image_info(self, image_bytes: bytes) -> Dict[str, Any]:
        """Get image information"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return {
                'format': image.format,
                'size': image.size,
                'mode': image.mode,
                'file_size': len(image_bytes),
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}
    
    def fix_image_orientation(self, image: Image.Image) -> Image.Image:
        """Fix image orientation based on EXIF data"""
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation] == 'Orientation':
                            if orientation in exif:
                                if exif[orientation] == 3:
                                    image = image.rotate(180, expand=True)
                                elif exif[orientation] == 6:
                                    image = image.rotate(270, expand=True)
                                elif exif[orientation] == 8:
                                    image = image.rotate(90, expand=True)
                            break
        except Exception as e:
            logger.warning(f"Could not fix orientation: {e}")
        
        return image
    
    def calculate_optimal_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """
        Calculate optimal dimensions for face detection
        - Maintain aspect ratio
        - Ensure minimum quality for face detection
        - Optimize for file size
        """
        # Don't upscale images
        if max(width, height) <= self.max_dimension:
            return width, height
        
        # Calculate scaling factor
        scale_factor = self.max_dimension / max(width, height)
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure minimum dimensions for face detection
        if min(new_width, new_height) < self.min_dimension:
            min_scale = self.min_dimension / min(width, height)
            new_width = int(width * min_scale)
            new_height = int(height * min_scale)
        
        # Round to even numbers for better compression
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        return new_width, new_height
    
    def optimize_for_face_detection(self, image: Image.Image) -> Image.Image:
        """Apply optimizations specifically for face detection"""
        
        # Convert to RGB if needed (face detection works best with RGB)
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
                image = background
            else:
                image = image.convert('RGB')
        
        # Apply subtle enhancement for better face detection
        # Note: Too much enhancement can hurt face detection accuracy
        try:
            from PIL import ImageEnhance
            
            # Subtle contrast enhancement (helps with low-contrast faces)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)  # Very subtle increase
            
            # Subtle sharpening (helps with slightly blurry faces)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.02)  # Very subtle increase
            
        except ImportError:
            logger.warning("PIL ImageEnhance not available, skipping enhancement")
        
        return image
    
    def compress_image(self, image_bytes: bytes, target_size: int = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress image while maintaining optimal quality for face detection
        
        Args:
            image_bytes: Original image bytes
            target_size: Target file size in bytes (default: self.max_file_size)
        
        Returns:
            Tuple of (compressed_bytes, compression_info)
        """
        if target_size is None:
            target_size = self.max_file_size
        
        original_size = len(image_bytes)
        
        # If already small enough, return as-is
        if original_size <= target_size:
            return image_bytes, {
                'original_size': original_size,
                'compressed_size': original_size,
                'compression_ratio': 1.0,
                'quality_used': 'original',
                'dimensions_changed': False
            }
        
        try:
            # Load and process image
            image = Image.open(io.BytesIO(image_bytes))
            original_dimensions = image.size
            
            # Fix orientation
            image = self.fix_image_orientation(image)
            
            # Calculate optimal dimensions
            new_width, new_height = self.calculate_optimal_dimensions(*image.size)
            dimensions_changed = (new_width, new_height) != image.size
            
            # Resize if needed
            if dimensions_changed:
                # Use high-quality resampling
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"Resized image: {original_dimensions} -> {(new_width, new_height)}")
            
            # Optimize for face detection
            image = self.optimize_for_face_detection(image)
            
            # Try different quality levels - BUT STOP if compression becomes too aggressive
            best_result = None
            min_acceptable_size = original_size * 0.3  # Never compress more than 70%
            
            for quality in self.quality_levels:
                output = io.BytesIO()
                
                # Save with premium optimization settings
                save_kwargs = {
                    'format': 'JPEG',
                    'quality': quality,
                    'optimize': True,
                    'progressive': True,  # Progressive JPEG for better loading
                    'subsampling': 0,     # 4:4:4 chroma subsampling for better quality
                    'dpi': (300, 300)     # High DPI for print quality
                }
                
                image.save(output, **save_kwargs)
                compressed_size = output.tell()
                
                compression_info = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compressed_size / original_size,
                    'quality_used': quality,
                    'dimensions_changed': dimensions_changed,
                    'original_dimensions': original_dimensions,
                    'new_dimensions': (new_width, new_height)
                }
                
                # SAFETY CHECK: Prevent over-compression
                if compressed_size < min_acceptable_size:
                    logger.warning(f"Compression too aggressive ({compressed_size} < {min_acceptable_size}), "
                                 f"stopping at previous quality level")
                    break
                
                # If we've reached target size with acceptable compression, use this result
                if compressed_size <= target_size:
                    logger.info(f"Compressed image: {original_size} -> {compressed_size} bytes "
                              f"(quality: {quality}, ratio: {compression_info['compression_ratio']:.2f})")
                    return output.getvalue(), compression_info
                
                # Keep track of best result
                best_result = (output.getvalue(), compression_info)
            
            # If we couldn't reach target size without over-compressing, return best result
            if best_result:
                logger.warning(f"Could not reach target size {target_size} without over-compression, "
                             f"best result: {best_result[1]['compressed_size']} bytes "
                             f"(ratio: {best_result[1]['compression_ratio']:.2f})")
                return best_result
            
            # If no acceptable compression found, try minimal resize with high quality
            if not dimensions_changed and max(original_dimensions) > 3000:
                logger.info("Trying minimal resize to avoid over-compression...")
                # Try reducing dimensions slightly while maintaining very high quality
                smaller_width = int(original_dimensions[0] * 0.9)
                smaller_height = int(original_dimensions[1] * 0.9)
                image = image.resize((smaller_width, smaller_height), Image.Resampling.LANCZOS)
                
                output = io.BytesIO()
                image.save(output, format='JPEG', quality=95, optimize=True, 
                          subsampling=0, progressive=True, dpi=(300, 300))
                
                compressed_size = output.tell()
                compression_info = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compressed_size / original_size,
                    'quality_used': '95 (with minimal resize)',
                    'dimensions_changed': True,
                    'original_dimensions': original_dimensions,
                    'new_dimensions': (smaller_width, smaller_height)
                }
                
                if compressed_size <= target_size:
                    return output.getvalue(), compression_info
            
            # Fallback: return original if compression failed
            return image_bytes, {
                'original_size': original_size,
                'compressed_size': original_size,
                'compression_ratio': 1.0,
                'quality_used': 'original',
                'dimensions_changed': False,
                'error': 'Compression failed'
            }
            
        except Exception as e:
            logger.error(f"Error compressing image: {e}")
            # Return original image if compression fails
            return image_bytes, {
                'original_size': original_size,
                'compressed_size': original_size,
                'compression_ratio': 1.0,
                'quality_used': 'original',
                'dimensions_changed': False,
                'error': str(e)
            }
    
    def validate_image(self, image_bytes: bytes) -> Tuple[bool, str]:
        """Validate if image is suitable for processing"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check format
            if image.format not in ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF']:
                return False, f"Unsupported format: {image.format}"
            
            # Check dimensions
            width, height = image.size
            if width < 100 or height < 100:
                return False, "Image too small (minimum 100x100 pixels)"
            
            if width > 10000 or height > 10000:
                return False, "Image too large (maximum 10000x10000 pixels)"
            
            # Check file size
            if len(image_bytes) > 50 * 1024 * 1024:  # 50MB
                return False, "File size too large (maximum 50MB)"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

# Global instance
image_processor = ImageProcessor()