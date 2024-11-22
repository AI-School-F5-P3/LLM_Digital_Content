# config.py
from pydantic import BaseModel
from typing import List, Dict

class Config:
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    MAX_LENGTH = 1000
    TEMPERATURE = 0.7 # Lower values generate more deterministic outputs and higher values more creative outputs

class ContentRequest(BaseModel):
    theme: str
    audience: str
    platform: str
    context: str = ""
    tone: str = "professional"