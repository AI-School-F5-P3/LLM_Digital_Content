# config.py
from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig:
    AVAILABLE_MODELS = {
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "Powerful general-purpose model",
            "generation_config": {
                "max_new_tokens": 256,
                "temperature": 0.4,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2
            }
        },
        "openai": {
            "name": "gpt-4o-mini",
            "description": "Advanced OpenAI language model"
        }
    }
    
    MAX_LENGTH = 512  # Reduced from 1000
    TEMPERATURE = 0.4
    GENERATION_TIMEOUT = 15  # Reduced from 60
    CACHE_DIR = "model_cache"

class ContentRequest(BaseModel):
    theme: str = Field(..., min_length=2, max_length=50)
    audience: str = Field(..., min_length=2, max_length=50)
    platform: str = Field(..., min_length=2, max_length=20)
    context: str = Field(default="", max_length=500)
    tone: str = Field(default="professional", patter="^(professional|casual|formal|friendly)$")
    company_info: Optional[str] = Field(default=None, max_length=300)
    selected_model: str = Field(default="mistral")
    include_image: bool = False