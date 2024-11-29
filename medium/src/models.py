# models.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import ModelConfig
import os
from dotenv import load_dotenv

load_dotenv()

# Deshabilitar completamente MPS en PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class ContentGenerator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.current_model = None
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
    def load_model(self, model_key: str):
        if model_key not in self.models:
            model_config = ModelConfig.AVAILABLE_MODELS[model_key]
            
            try:
                # Cargar tokenizador
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config["name"],
                    token=self.huggingface_token,
                    cache_dir=ModelConfig.CACHE_DIR,
                    padding_side='left'
                )
                
                # Añadir token de padding si no existe
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Configuración específica para MPS
                model = AutoModelForCausalLM.from_pretrained(
                    model_config["name"],
                    token=self.huggingface_token,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    cache_dir=ModelConfig.CACHE_DIR
                )
                
                # Mover a CPU forzado
                device = torch.device("cpu")
                model = model.to(device)
                
                self.models[model_key] = model
                self.tokenizers[model_key] = tokenizer
                
            except Exception as e:
                print(f"Error loading {model_key} model: {e}")
                raise
        
        return self.models[model_key], self.tokenizers[model_key]
    
    def generate_content(self, prompt: str, model_key: str = "mistral") -> dict:
        try:
            model, tokenizer = self.load_model(model_key)
            
            device = torch.device("cpu")  # Asegurar CPU
            
            # Preparar input
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512,
                truncation=True, 
                padding=True
            ).to(device)
            
            # Generar con restricciones
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.4,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2
                )
            
            # Decodificar output completo
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Eliminar el prompt original de manera más robusta
            generated_text = full_text[len(prompt):].strip()
            
            # Si aún contiene el prompt, cortar manualmente
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Validar contenido
            if len(generated_text.split()) < 30:
                return {
                    "status": "error",
                    "content": "No se generó contenido suficiente.",
                    "model_used": model_key
                }
            
            return {
                "status": "success",
                "content": generated_text,
                "model_used": model_key
            }
        
        except Exception as e:
            return {
                "status": "error",
                "content": f"Error de generación: {str(e)}",
                "model_used": model_key
            }
    
    def __del__(self):
        # Limpiar memoria
        for model in self.models.values():
            del model
        self.models.clear()
        
        # Limpiar caché de MPS si está disponible
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

# Optional: Add to your config.py to support these settings
ModelConfig.AVAILABLE_MODELS = {
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",  # Consider using a smaller variant
        "description": "Compact, efficient language model"
    }
}