# backend/cdn_service.py
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from typing import Dict, List, Optional, Tuple
import hashlib
from urllib.parse import urlparse

class CDNService:
    def __init__(self):
        # Cloudinary is already configured in config.py
        self.transformations = {
            'thumbnail': {
                'width': 150,
                'height': 150,
                'crop': 'fill',
                'gravity': 'face',
                'quality': 'auto:low',
                'format': 'webp'
            },
            'preview': {
                'width': 400,
                'height': 300,
                'crop': 'fill',
                'gravity': 'faces',
                'quality': 'auto:good',
                'format': 'webp'
            },
            'display': {
                'width': 800,
                'height': 600,
                'crop': 'limit',
                'quality': 'auto:good',
                'format': 'auto'
            },
            'full': {
                'width': 1920,
                'height': 1080,
                'crop': 'limit',
                'quality': 'auto:best',
                'format': 'auto'
            },
            'face_crop': {
                'width': 200,
                'height': 200,
                'crop': 'thumb',
                'gravity': 'face',
                'zoom': 0.7,
                'format': 'jpg'
            }
        }
    
    def get_optimized_url(self, original_url: str, 
                          transformation: str = 'display') -> str:
        """Get optimized image URL with transformation"""
        try:
            # Extract public_id from Cloudinary URL
            public_id = self.extract_public_id(original_url)
            if not public_id:
                return original_url
            
            # Get transformation settings
            transform = self.transformations.get(transformation, {})
            
            # Generate optimized URL
            url, _ = cloudinary_url(
                public_id,
                transformation=[transform],
                secure=True,
                cdn_subdomain=True,  # Use multiple CDN subdomains
                responsive=True,     # Support responsive images
                dpr='auto'          # Auto device pixel ratio
            )
            
            return url
        except:
            return original_url
    
    def get_responsive_urls(self, original_url: str) -> Dict[str, str]:
        """Get multiple sizes for responsive design"""
        try:
            public_id = self.extract_public_id(original_url)
            if not public_id:
                return {'original': original_url}
            
            urls = {}
            breakpoints = [
                ('mobile', {'width': 375, 'quality': 'auto:low'}),
                ('tablet', {'width': 768, 'quality': 'auto:good'}),
                ('desktop', {'width': 1280, 'quality': 'auto:good'}),
                ('retina', {'width': 2560, 'quality': 'auto:best'})
            ]
            
            for name, transform in breakpoints:
                url, _ = cloudinary_url(
                    public_id,
                    transformation=[{
                        **transform,
                        'crop': 'limit',
                        'format': 'auto'
                    }],
                    secure=True
                )
                urls[name] = url
            
            urls['original'] = original_url
            return urls
            
        except:
            return {'original': original_url}
    
    def get_face_thumbnails(self, original_url: str, 
                           face_coordinates: List[Dict]) -> List[str]:
        """Get cropped thumbnails for each face"""
        thumbnails = []
        
        try:
            public_id = self.extract_public_id(original_url)
            if not public_id:
                return thumbnails
            
            for i, face in enumerate(face_coordinates[:5]):  # Max 5 faces
                # Convert bbox to Cloudinary crop coordinates
                x = face.get('x', 0)
                y = face.get('y', 0)
                w = face.get('w', 100)
                h = face.get('h', 100)
                
                # Generate face crop URL
                url, _ = cloudinary_url(
                    public_id,
                    transformation=[
                        {
                            'crop': 'crop',
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h
                        },
                        {
                            'width': 150,
                            'height': 150,
                            'crop': 'fill',
                            'quality': 'auto:good',
                            'format': 'jpg'
                        }
                    ],
                    secure=True
                )
                thumbnails.append(url)
            
            return thumbnails
            
        except:
            return thumbnails
    
    def generate_blur_url(self, original_url: str, 
                         blur_level: int = 1000) -> str:
        """Generate blurred version for privacy"""
        try:
            public_id = self.extract_public_id(original_url)
            if not public_id:
                return original_url
            
            url, _ = cloudinary_url(
                public_id,
                transformation=[{
                    'effect': f'blur:{blur_level}',
                    'quality': 'auto:low'
                }],
                secure=True
            )
            
            return url
        except:
            return original_url
    
    def generate_placeholder(self, original_url: str) -> str:
        """Generate tiny placeholder for lazy loading"""
        try:
            public_id = self.extract_public_id(original_url)
            if not public_id:
                return original_url
            
            # Tiny base64-encoded placeholder
            url, _ = cloudinary_url(
                public_id,
                transformation=[{
                    'width': 30,
                    'quality': 10,
                    'format': 'webp',
                    'effect': 'blur:300'
                }],
                secure=True
            )
            
            return url
        except:
            return original_url
    
    def extract_public_id(self, cloudinary_url: str) -> Optional[str]:
        """Extract public_id from Cloudinary URL"""
        try:
            # Parse URL to get path
            parsed = urlparse(cloudinary_url)
            path_parts = parsed.path.split('/')
            
            # Find upload index
            if 'upload' in path_parts:
                upload_idx = path_parts.index('upload')
                # Public ID is after version (v...) or directly after upload
                if upload_idx < len(path_parts) - 1:
                    remaining = path_parts[upload_idx + 1:]
                    # Skip version if present
                    if remaining[0].startswith('v'):
                        remaining = remaining[1:]
                    # Join remaining parts and remove extension
                    public_id = '/'.join(remaining)
                    # Remove file extension
                    if '.' in public_id:
                        public_id = public_id.rsplit('.', 1)[0]
                    return public_id
        except:
            pass
        return None
    
    def batch_optimize_urls(self, urls: List[str], 
                           transformation: str = 'preview') -> List[str]:
        """Batch optimize multiple URLs"""
        return [
            self.get_optimized_url(url, transformation) 
            for url in urls
        ]
    
    def get_download_url(self, original_url: str, 
                        filename: Optional[str] = None) -> str:
        """Get download URL with custom filename"""
        try:
            public_id = self.extract_public_id(original_url)
            if not public_id:
                return original_url
            
            attachment = f"attachment:{filename}" if filename else "attachment"
            
            url, _ = cloudinary_url(
                public_id,
                transformation=[{
                    'flags': attachment,
                    'quality': 'auto:best'
                }],
                secure=True
            )
            
            return url
        except:
            return original_url

# Singleton instance
cdn_service = CDNService()