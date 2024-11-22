# config.py
from pydantic import BaseModel
from typing import List, Dict

class Config:
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    MAX_LENGTH = 1000
    TEMPERATURE = 0.7

class ContentRequest(BaseModel):
    theme: str
    audience: str
    platform: str
    context: str = ""
    tone: str = "professional"