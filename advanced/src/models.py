# src/models.py
# Standard library imports
import os
import time
from typing import Dict, Any

# Environment and configuration
from dotenv import load_dotenv

# Machine learning and AI libraries
import torch
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM

# Third-party libraries
import requests # for finantial news api
import finnhub
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

# Local imports
from config import ModelConfig

load_dotenv()

class ContentGenerator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.current_model = None
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        self.rag_storage = ScientificRAG() # Existing initialization...
        
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
    
    def generate_financial_news(self, language='en'):
        """
        Fetch updated financial market information using Finnhub API
        """
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
                language = prompt.split("Language:")[1].split("\n")[0].strip() if "Language:" in prompt else "en"
                
                messages = [
                    {"role": "system", "content": f"You must respond in {language}. Generate content precisely following the user's requirements."},
                    {"role": "user", "content": prompt}
                ]
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=512,
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
            
            # Use model-specific generation config
            generation_config = ModelConfig.AVAILABLE_MODELS[model_key]['generation_config']
            
            # Use the config in model generation
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
            
    def generate_scientific_content(self, theme: str, language: str = 'en'):
        """
        Advanced RAG-based scientific content generation
        Uses arXiv API with more sophisticated retrieval and processing
        """
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

            # Add documents to RAG storage
            self.rag_storage.add_documents(documents)
            
            # Perform semantic search on the theme
            semantic_search_results = self.rag_storage.search_documents(theme)
            
            # Combine the original context with semantic search results
            scientific_context = {
                "theme": theme,
                "arxiv_documents": [
                    {
                        "title": doc['title'],
                        "summary": doc['summary'],
                        "authors": doc['authors']
                    } for doc in documents
                ],
                "semantic_search_results": semantic_search_results,
                "language": language
            }

            return scientific_context

        except Exception as e:
            print(f"Scientific content error: {e}")  # Add detailed logging
            return {
                "status": "error",
                "message": f"Scientific content retrieval error: {str(e)}",
                "details": str(e)  # More detailed error
            }
    
    def __del__(self):
        # Limpiar memoria
        for model in self.models.values():
            del model
        self.models.clear()
        
        # Limpiar caché de MPS si está disponible
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
class ScientificRAG:
    def __init__(self, cache_dir='vector_cache'):
        # Create cache directory if it doesn't exist
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=self.cache_dir)
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(name="scientific_documents")
        
        # Embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def add_documents(self, documents):
        for doc in documents:
            # Generate unique ID and embedding
            doc_id = str(hash(doc['title']))
            embedding = self.model.encode(doc['summary']).tolist()
            
            # Add to Chroma collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[{
                    'title': doc['title'],
                    'authors': ', '.join(doc['authors']),
                    'theme': doc.get('theme', 'general')
                }],
                documents=[doc['summary']]
            )

    def search_documents(self, query, top_k=3):
        query_embedding = self.model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Transform results into a more readable format
        formatted_results = []
        for i in range(top_k):
            formatted_results.append({
                'title': results['metadatas'][0][i]['title'],
                'summary': results['documents'][0][i],
                'authors': results['metadatas'][0][i]['title'],
                'relevance_score': results['distances'][0][i]
            })
        
        return formatted_results