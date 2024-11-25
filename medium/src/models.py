# models.py
import torch
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
        
    def load_model(self, model_key: str):
        if model_key not in self.models:
            model_config = ModelConfig.AVAILABLE_MODELS[model_key]
            
            # Load tokenizer
            self.tokenizers[model_key] = AutoTokenizer.from_pretrained(
                model_config["name"],
                token=self.huggingface_token,
                cache_dir=ModelConfig.CACHE_DIR
            )
            
            # Load model
            self.models[model_key] = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                token=self.huggingface_token,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                cache_dir=ModelConfig.CACHE_DIR
            )
            
            self.models[model_key].eval()
        
        self.current_model = model_key
        return self.models[model_key], self.tokenizers[model_key]
    
    def generate_content(self, prompt: str, model_key: str = "mistral") -> dict:
        try:
            model, tokenizer = self.load_model(model_key)
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=ModelConfig.MAX_LENGTH,
                    temperature=ModelConfig.TEMPERATURE,
                    do_sample=True,
                    num_beams=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    max_time=ModelConfig.GENERATION_TIMEOUT
                )
            
            generated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return {
                "status": "success",
                "content": generated_text.strip(),
                "model_used": model_key
            }
            
        except Exception as e:
            return {
                "status": "error",
                "content": f"Error generating content: {str(e)}",
                "model_used": model_key
            }
    
    def __del__(self):
        for model in self.models.values():
            del model
        self.models.clear()
        self.tokenizers.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None