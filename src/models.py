# src/models.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config
import logging
import os
from dotenv import load_dotenv
import sentencepiece

# Load environment variables from .env file
load_dotenv()

class ContentGenerator:
    def __init__(self):
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            use_auth_token=huggingface_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            use_auth_token=huggingface_token, # Use Hugging Face authentication token for private models
            torch_dtype=torch.float16, # Use float16 for faster generation, to reduce memory usage as they are half the size of float32, and to take advantage of the mixed precision training.
            device_map="cpu", # My graphic card intel integrated graphics is not compatible with CUDA, so I'm using CPU directly.
            low_cpu_mem_usage=True  # It optimizes the memory usage of the model by reducing the memory usage of the model's weights.
        )
        
    def generate_content(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=Config.MAX_LENGTH, # Maximum length of the generated content
                temperature=Config.TEMPERATURE, # Controls the randomness of the generated content
                do_sample=True, # Enable sampling (i.e., non-deterministic generation)
                pad_token_id=self.tokenizer.eos_token_id # End of sequence token (EOS). Asigna el end-of-sequence token (EOS) como el token de relleno (pad_token_id). Esto significa que cualquier "relleno" necesario al final de la secuencia será completado con el token EOS.
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True) # special tokens are symbols like [CLS], [SEP], [PAD], etc. that are used by the tokenizer to understand the structure of the input text, adn with this method we are removing them.
        except Exception as e:
            logging.error(f"Error generating content: {str(e)}")
            return "Error generating content. Please try again."