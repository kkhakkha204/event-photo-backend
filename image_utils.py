# backend/image_utils.py - Quality Preservation Version
from PIL import Image, ImageOps, ExifTags
import io
import os
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Quality-focused image processing - preserves near-original quality
    """
    
    def __init__(self):
        self.max_file_size = 9.5 * 1024 * 1024  # 9.5MB (safe under 10MB limit)
        self.compression_trigger = 9 * 1024 * 1024  # Only compress if > 9MB
        self.max_dimension = 3840  # 4K support - preserve original size
        self.min_dimension = 640   
        
        # High-quality levels - start from near-lossless
        self.quality_levels = [99, 97, 95, 93, 90, 87, 84]
        
        # Safety limits
        self.min_compression_ratio = 0.3  # Never compress below 30% of original
        self.preferred_compression_ratio = 0.7  # Target 70% (conservative)
    
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
    
    def calculate_conservative_dimensions(self, width: int, height: int, force_resize: bool = False) -> Tuple[int, int]:
        """
        Conservative dimension calculation - preserve original size when possible
        """
        # Keep original dimensions if within 4K limit and not forced
        if not force_resize and max(width, height) <= self.max_dimension:
            return width, height
        
        # Only resize if absolutely necessary
        if max(width, height) > self.max_dimension:
            scale_factor = self.max_dimension / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        elif force_resize:
            # Minimal resize (90% of original) as safety fallback
            new_width = int(width * 0.9)
            new_height = int(height * 0.9)
        else:
            return width, height
        
        # Ensure minimum dimensions
        if min(new_width, new_height) < self.min_dimension:
            min_scale = self.min_dimension / min(width, height)
            new_width = int(width * min_scale)
            new_height = int(height * min_scale)
        
        # Round to even numbers for better compression
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        return new_width, new_height
    
    def optimize_for_face_detection(self, image: Image.Image) -> Image.Image:
        """Apply minimal optimizations for face detection while preserving quality"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
                image = background
            else:
                image = image.convert('RGB')
        
        # NO enhancement - keep original quality
        # Face detection works fine with original images
        
        return image
    
    def compress_image(self, image_bytes: bytes, target_size: int = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Quality-focused compression - preserves near-original quality
        
        Smart Logic:
        - < 9MB: No compression (original quality)
        - 9-12MB: Quality reduction only (99% → 87%)  
        - > 12MB: Minimal resize (90%) + high quality (95%)
        - Safety: Never compress below 30% of original
        """
        if target_size is None:
            target_size = self.max_file_size
        
        original_size = len(image_bytes)
        
        # RULE 1: If already under trigger threshold, return as-is
        if original_size <= self.compression_trigger:
            logger.info(f"Image size {original_size / (1024*1024):.1f}MB < 9MB - keeping original quality")
            return image_bytes, {
                'original_size': original_size,
                'compressed_size': original_size,
                'compression_ratio': 1.0,
                'quality_used': 'original',
                'dimensions_changed': False,
                'strategy': 'no_compression_needed'
            }
        
        try:
            # Load and process image
            image = Image.open(io.BytesIO(image_bytes))
            original_dimensions = image.size
            
            # Fix orientation
            image = self.fix_image_orientation(image)
            
            # Optimize for face detection
            image = self.optimize_for_face_detection(image)
            
            # Calculate safety limits
            min_acceptable_size = original_size * self.min_compression_ratio  # 30% minimum
            preferred_target = original_size * self.preferred_compression_ratio  # 70% preferred
            
            logger.info(f"Compressing {original_size / (1024*1024):.1f}MB image")
            logger.info(f"Targets: preferred={preferred_target / (1024*1024):.1f}MB, "
                       f"max={target_size / (1024*1024):.1f}MB, "
                       f"min_safe={min_acceptable_size / (1024*1024):.1f}MB")
            
            # STRATEGY 1: Quality reduction only (9-12MB range)
            if original_size <= 12 * 1024 * 1024:  # <= 12MB
                logger.info("Strategy: Quality reduction only (preserving dimensions)")
                
                for quality in self.quality_levels:
                    output = io.BytesIO()
                    
                    # High-quality JPEG settings
                    save_kwargs = {
                        'format': 'JPEG',
                        'quality': quality,
                        'optimize': True,
                        'progressive': True,
                        'subsampling': 0,  # 4:4:4 chroma subsampling (best quality)
                        'dpi': (300, 300)
                    }
                    
                    image.save(output, **save_kwargs)
                    compressed_size = output.tell()
                    
                    # Safety check
                    if compressed_size < min_acceptable_size:
                        logger.warning(f"Quality {quality}% would over-compress "
                                     f"({compressed_size / (1024*1024):.1f}MB), stopping")
                        break
                    
                    compression_info = {
                        'original_size': original_size,
                        'compressed_size': compressed_size,
                        'compression_ratio': compressed_size / original_size,
                        'quality_used': f'{quality}%',
                        'dimensions_changed': False,
                        'original_dimensions': original_dimensions,
                        'new_dimensions': original_dimensions,
                        'strategy': 'quality_reduction_only'
                    }
                    
                    # Success if under target
                    if compressed_size <= target_size:
                        logger.info(f"✓ Success: {original_size / (1024*1024):.1f}MB → "
                                  f"{compressed_size / (1024*1024):.1f}MB "
                                  f"(quality: {quality}%, ratio: {compression_info['compression_ratio']:.2f})")
                        return output.getvalue(), compression_info
                    
                    # If we hit preferred target, use this
                    if compressed_size <= preferred_target:
                        logger.info(f"✓ Preferred target hit: {compressed_size / (1024*1024):.1f}MB "
                                  f"(quality: {quality}%)")
                        return output.getvalue(), compression_info
            
            # STRATEGY 2: Minimal resize + high quality (> 12MB)
            logger.info("Strategy: Minimal resize (90%) + high quality (95%)")
            
            # Calculate minimal resize dimensions
            new_width, new_height = self.calculate_conservative_dimensions(
                *original_dimensions, 
                force_resize=True
            )
            
            # Resize with high-quality resampling
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try with high quality
            output = io.BytesIO()
            save_kwargs = {
                'format': 'JPEG',
                'quality': 95,  # High quality
                'optimize': True,
                'progressive': True,
                'subsampling': 0,
                'dpi': (300, 300)
            }
            
            resized_image.save(output, **save_kwargs)
            compressed_size = output.tell()
            
            compression_info = {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / original_size,
                'quality_used': '95% (with minimal resize)',
                'dimensions_changed': True,
                'original_dimensions': original_dimensions,
                'new_dimensions': (new_width, new_height),
                'strategy': 'minimal_resize_high_quality'
            }
            
            # Check if minimal resize worked
            if compressed_size <= target_size and compressed_size >= min_acceptable_size:
                logger.info(f"✓ Minimal resize success: {original_size / (1024*1024):.1f}MB → "
                          f"{compressed_size / (1024*1024):.1f}MB "
                          f"(95% quality, {original_dimensions} → {(new_width, new_height)})")
                return output.getvalue(), compression_info
            
            # STRATEGY 3: If minimal resize still too big, try quality reduction on resized image
            if compressed_size > target_size:
                logger.info("Strategy: Quality reduction on resized image")
                
                for quality in [90, 87, 84]:
                    output = io.BytesIO()
                    save_kwargs['quality'] = quality
                    
                    resized_image.save(output, **save_kwargs)
                    compressed_size = output.tell()
                    
                    if compressed_size < min_acceptable_size:
                        logger.warning(f"Quality {quality}% on resized would over-compress, stopping")
                        break
                    
                    compression_info.update({
                        'compressed_size': compressed_size,
                        'compression_ratio': compressed_size / original_size,
                        'quality_used': f'{quality}% (with minimal resize)',
                        'strategy': 'resize_plus_quality_reduction'
                    })
                    
                    if compressed_size <= target_size:
                        logger.info(f"✓ Resize + quality success: {original_size / (1024*1024):.1f}MB → "
                                  f"{compressed_size / (1024*1024):.1f}MB "
                                  f"(quality: {quality}%)")
                        return output.getvalue(), compression_info
            
            # FALLBACK: Return best attempt (even if over target)
            logger.warning(f"Could not reach target {target_size / (1024*1024):.1f}MB safely. "
                         f"Best result: {compressed_size / (1024*1024):.1f}MB "
                         f"(ratio: {compression_info['compression_ratio']:.2f})")
            
            return output.getvalue(), compression_info
            
        except Exception as e:
            logger.error(f"Error compressing image: {e}")
            # Return original image if compression fails
            return image_bytes, {
                'original_size': original_size,
                'compressed_size': original_size,
                'compression_ratio': 1.0,
                'quality_used': 'original',
                'dimensions_changed': False,
                'error': str(e),
                'strategy': 'fallback_original'
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
            
            # Check file size (increased limit for quality preservation)
            if len(image_bytes) > 100 * 1024 * 1024:  # 100MB
                return False, "File size too large (maximum 100MB)"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image: {str(e)}"

# Global instance
image_processor = ImageProcessor()