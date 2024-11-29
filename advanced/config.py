import os
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    AVAILABLE_MODELS = {
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "Powerful general-purpose model"
        },
        "llama2": {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "description": "Optimized for conversational tasks"
        },
        "openai": {
            "name": "gpt-3.5-turbo",
            "description": "Advanced OpenAI language model"
        }
    }
    
    MAX_LENGTH = 512
    TEMPERATURE = 0.4
    GENERATION_TIMEOUT = 15
    CACHE_DIR = "model_cache"
    
    # Advanced language configuration
    SUPPORTED_LANGUAGES = {
        "es": "Spanish",
        "en": "English", 
        "fr": "French",
        "it": "Italian"
    }

class ContentRequest(BaseModel):
    theme: str
    audience: str
    platform: str
    context: str = ""
    tone: str = "professional"
    company_info: Optional[str] = None
    selected_model: str = "mistral"
    include_image: bool = False
    language: str = "en"  # Default to English