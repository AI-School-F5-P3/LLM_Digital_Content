# src/models.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config
import logging

class ContentGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate_content(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=Config.MAX_LENGTH,
                temperature=Config.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error generating content: {str(e)}")
            return "Error generating content. Please try again."