# src/image_utils.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class ImageRetriever:
    def __init__(self):
        self.unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
        self.pixabay_api_key = os.getenv("PIXABAY_API_KEY")
        self.unsplash_base_url = "https://api.unsplash.com/search/photos"
        self.pixabay_base_url = "https://pixabay.com/api/"
    
    def get_relevant_image(self, theme, count=1):
        """
        Retrieve relevant images from Unsplash based on theme
        
        Args:
            theme (str): Topic/theme to search images for
            count (int): Number of images to retrieve
        
        Returns:
            list: URLs of retrieved images
        """
        if not self.unsplash_access_key:
            raise ValueError("Unsplash API key not configured")
        
        params = {
            "query": theme,
            "client_id": self.unsplash_access_key,
            "per_page": count,
            "orientation": "squarish"
        }
        
        try:
            response = requests.get(self.unsplash_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            images = [
                {
                    "url": photo["urls"]["regular"],
                    "description": photo.get("description", theme),
                    "alt_description": photo.get("alt_description", theme)
                } 
                for photo in data.get("results", [])
            ]
            
            return images
        except requests.RequestException as e:
            print(f"Error retrieving images: {e}")
            return []
        
    def get_pixabay_images(self, theme, count=1):
        """
        Retrieve relevant images from Pixabay based on theme
        """
        if not self.pixabay_api_key:
            raise ValueError("Pixabay API key not configured")
        
        params = {
            "key": self.pixabay_api_key,
            "q": theme,
            "image_type": "photo",
            "per_page": count
        }
        
        try:
            response = requests.get(self.pixabay_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            images = [
                {
                    "url": hit["webformatURL"],
                    "tags": hit.get("tags", ""),
                    "user": hit.get("user", "")
                }
                for hit in data.get("hits", [])
            ]
            
            return images
        except requests.RequestException as e:
            print(f"Error retrieving images from Pixabay: {e}")
            return []