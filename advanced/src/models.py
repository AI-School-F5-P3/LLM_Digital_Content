# models.py
import torch
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import ModelConfig
import os
from dotenv import load_dotenv
import requests  # For financial news API

load_dotenv()

class ContentGenerator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.current_model = None
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        
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
    
    def generate_financial_news(self, language='es'):
        """
        Fetch updated financial market information
        Uses a hypothetical financial news API
        """
        try:
            # Replace with actual financial news API endpoint
            api_url = "https://financial-news-api.example.com/latest"
            params = {"language": language}
            
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_financial_news(self, language='en'):
        """
        Fetch updated financial market information using Finnhub API
        """
        import finnhub

        try:
            # Load Finnhub API key from environment
            finnhub_api_key = os.getenv("FINNHUB_API_KEY")
            if not finnhub_api_key:
                return {"error": "Finnhub API key not configured"}

            # Initialize Finnhub client
            finnhub_client = finnhub.Client(api_key=finnhub_api_key)

            # Fetch market news
            news = finnhub_client.general_news('general', min_id=0)

            # Process and filter news
            processed_news = []
            for article in news[:5]:  # Limit to 5 recent articles
                processed_news.append({
                    "headline": article.get('headline', ''),
                    "summary": article.get('summary', ''),
                    "source": article.get('source', ''),
                    "datetime": article.get('datetime', '')
                })

            return {
                "status": "success",
                "news": processed_news,
                "language": language
            }

        except Exception as e:
            return {
                "status": "error", 
                "message": f"Financial news retrieval error: {str(e)}"
            }
    
    def generate_content(self, prompt: str, model_key: str = "mistral") -> dict:
        try:
            if model_key == "openai":
                # OpenAI GPT generation
                if not self.openai_api_key:
                    return {
                        "status": "error",
                        "content": "OpenAI API key not configured",
                        "model_used": model_key
                    }
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful content generation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=256,
                    temperature=0.4
                )
                
                generated_text = response.choices[0].message.content.strip()
                
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
            
    def generate_scientific_content(self, theme: str, language: str = 'en'):
        """
        Advanced RAG-based scientific content generation
        Uses arXiv API with more sophisticated retrieval and processing
        """
        import requests
        import xml.etree.ElementTree as ET
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            # ArXiv API query
            base_url = "http://export.arxiv.org/api/query"
            query_params = {
                "search_query": f"all:{theme}",
                "start": 0,
                "max_results": 10,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(base_url, params=query_params)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.text)
            namespace = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

            # Extract relevant scientific documents
            documents = []
            for entry in root.findall('atom:entry', namespace):
                title = entry.find('atom:title', namespace).text
                summary = entry.find('atom:summary', namespace).text
                authors = [author.find('atom:name', namespace).text 
                        for author in entry.findall('atom:author', namespace)]
                
                documents.append({
                    'title': title,
                    'summary': summary,
                    'authors': authors
                })

            # Semantic similarity ranking using sentence transformers
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode theme and documents
            theme_embedding = model.encode([theme])
            document_embeddings = model.encode([doc['summary'] for doc in documents])
            
            # Compute similarity scores
            similarities = cosine_similarity(theme_embedding, document_embeddings)[0]
            
            # Rank and select top documents
            ranked_docs = sorted(
                zip(documents, similarities), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]

            # Create structured scientific context
            scientific_context = {
                "theme": theme,
                "top_documents": [
                    {
                        "title": doc['title'],
                        "summary": doc['summary'],
                        "authors": doc['authors'],
                        "relevance_score": score
                    } for doc, score in ranked_docs
                ],
                "language": language
            }

            return scientific_context

        except Exception as e:
            return {
                "status": "error",
                "message": f"Scientific content retrieval error: {str(e)}"
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