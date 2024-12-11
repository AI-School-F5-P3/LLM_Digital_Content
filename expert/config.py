# config.py
import os
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

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
        "llama2": {
            "name": "meta-llama/Llama-2-7b-chat-hf",
            "description": "Optimized for conversational tasks",
            "generation_config": {
                "max_new_tokens": 256,
                "temperature": 0.5,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2
            }
        },
        "openai": {
            "name": "gpt-4o-mini",
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
    
    @classmethod
    def validate_model(cls, model_key: str) -> bool:
        """
        Validate if the selected model is available
        """
        return model_key in cls.AVAILABLE_MODELS

class ContentRequest(BaseModel):
    theme: str = Field(..., min_length=2, max_length=50)
    audience: str = Field(..., min_length=2, max_length=50)
    platform: str = Field(..., min_length=2, max_length=20)
    context: str = Field(default="", max_length=500)
    tone: str = Field(default="professional", patter="^(professional|casual|formal|friendly)$")
    company_info: Optional[str] = Field(default=None, max_length=300)
    selected_model: str = Field(default="mistral")
    include_image: bool = False
    language: str = Field(default="en", patter="^(es|en|fr|it)$")
    
    @classmethod
    def validate_model(cls, values):
        if not ModelConfig.validate_model(values.get('selected_model', 'mistral')):
            raise ValueError("Invalid model selected")