# models.py
import torch
from openai import OpenAI
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import ModelConfig
import os
from dotenv import load_dotenv

load_dotenv()

class ContentGenerator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.current_model = None
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    def load_model(self, model_key: str):
        if model_key == "openai":
            return None, None  # OpenAI doesn't use traditional loading
        
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
                
                # Model loading
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
            # Extract company info from prompt if it exists
            company_info = ""
            if "{company_info}" in prompt:
                try:
                    # Split the prompt and extract company info
                    company_info = prompt.split("{company_info}")[1].split("{")[0].strip()
                except Exception as e:
                    print(f"Error extracting company info: {e}")
            
            # Remove {company_info} placeholder from prompt
            prompt = prompt.replace("{company_info}", "").strip()

            # Add company info to the prompt if available
            if company_info:
                prompt += f"\n\nAdditional Context - Company/Personal Info: {company_info}"
                
            if model_key == "openai":
                client = OpenAI(api_key=self.openai_api_key)
                
                # Truncate prompt if too long
                max_prompt_length = 4000
                if len(prompt) > max_prompt_length:
                    prompt = prompt[:max_prompt_length]
                    
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"Generate content precisely following the user's requirements."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.4
                )
                
                generated_text = response.choices[0].message.content.strip()
                generated_text = generated_text.replace(prompt, '').strip()
                
                return {
                    "status": "success",
                    "content": generated_text,
                    "model_used": model_key
                }
                
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
            
            # Use model-specific generation config
            generation_config = ModelConfig.AVAILABLE_MODELS[model_key]['generation_config']
            
            # Generar con restricciones
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=generation_config['max_new_tokens'],
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=generation_config['temperature'],
                    top_k=generation_config['top_k'],
                    top_p=generation_config['top_p'],
                    repetition_penalty=generation_config['repetition_penalty'],
                    no_repeat_ngram_size=generation_config['no_repeat_ngram_size']
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