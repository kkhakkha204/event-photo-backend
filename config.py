import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def upload_image(file):
    """Upload image to Cloudinary"""
    result = cloudinary.uploader.upload(file)
    return result['secure_url']