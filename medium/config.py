# config.py
from pydantic import BaseModel
from typing import Optional

class ModelConfig:
    AVAILABLE_MODELS = {
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.3",
            "description": "Powerful general-purpose model"
        },
        "llama2": {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "description": "Optimized for conversational tasks"
        }
    }
    
    MAX_LENGTH = 1000
    TEMPERATURE = 0.7
    GENERATION_TIMEOUT = 60  # Increased timeout
    CACHE_DIR = "model_cache"

class ContentRequest(BaseModel):
    theme: str
    audience: str
    platform: str
    context: str = ""
    tone: str = "professional"
    company_info: Optional[str] = None
    selected_model: str = "mistral"
    include_image: bool = False