# config.py
from pydantic import BaseModel

class Config:
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    MAX_LENGTH = 1500  # Reducido para mejor rendimiento
    TEMPERATURE = 0.7
    CACHE_DIR = "model_cache"  # Directorio para caché
    MIN_LENGTH = 50
    TOP_K = 50 # Selecciona las 50 palabras más probables
    TOP_P = 0.9 # Selecciona las palabras hasta que la suma de las probabilidades sea 0.9. Basado en la probabilidad acumulada
    GENERATION_TIMEOUT = 30  # Límite de tiempo en segundos

class ContentRequest(BaseModel):
    theme: str
    audience: str
    platform: str
    context: str = ""
    tone: str = "professional"