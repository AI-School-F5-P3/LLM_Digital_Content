# src/models.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config
import logging
import os
from dotenv import load_dotenv

load_dotenv()

class ContentGenerator:
    def __init__(self):
        huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Configuración del tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            token=huggingface_token,
            cache_dir="model_cache"  # Cachear el modelo para futuras cargas
        )
        
        # Configuración optimizada del modelo
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            token=huggingface_token,
            torch_dtype=torch.float32,  # Cambiado a float32 para mejor compatibilidad con CPU
            device_map="cpu",
            low_cpu_mem_usage=True,
            cache_dir="model_cache",
            max_memory={0: "4GB"},  # Limitar el uso de memoria
            offload_folder="offload"  # Carpeta para offload de tensores
        )
        
        # Configurar el modelo en modo evaluación
        self.model.eval()
    
    def generate_content(self, prompt: str) -> str:
        try:
            # Tokenización con manejo de longitud máxima
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,  # Limitar la longitud de entrada
                truncation=True
            )
            
            # Configuración más conservadora para la generación
            with torch.no_grad():  # Desactivar gradientes para ahorrar memoria
                outputs = self.model.generate(
                    **inputs,
                    max_length=Config.MAX_LENGTH,
                    min_length=50,  # Asegurar una longitud mínima
                    temperature=Config.TEMPERATURE,
                    do_sample=True,
                    num_beams=1,  # Usar generación simple en lugar de beam search
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,  # Evitar repeticiones
                    max_time=Config.GENERATION_TIMEOUT  # Límite de tiempo en segundos
                )
            
            # Decodificar la salida
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logging.error(f"Error generating content: {str(e)}")
            return f"Error generating content: {str(e)}"

    def __del__(self):
        # Limpieza de memoria al destruir la instancia
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None