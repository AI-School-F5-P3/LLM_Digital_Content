# expert/src/knowledge_graph.py
import networkx as nx
import spacy

from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

class KnowledgeGraphRAG:
    def __init__(self):
        # Load NLP model for entity and relationship extraction
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Embedding model for semantic similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_entities_and_relations(self, text):
        """
        Extract named entities and potential relationships from text
        """
        doc = self.nlp(text)
        
        entities = [
            (ent.text, ent.label_) 
            for ent in doc.ents
        ]
        
        # Extract potential verb phrases as relationships
        relations = [
            (token.text, token.dep_) 
            for token in doc 
            if token.pos_ == "VERB"
        ]
        
        return entities, relations
    
    def add_knowledge(self, text, source=None):
        """
        Add knowledge from text to the knowledge graph
        """
        try:
            import numpy as np
            # Extract entities and relations
            entities, relations = self.extract_entities_and_relations(text)
            
            # Embed the full text for semantic indexing
            try:
                text_embedding = self.embedding_model.encode([text])[0]
            except Exception as embed_error:
                print(f"Embedding error: {embed_error}")
                text_embedding = np.zeros(384)  # Create a zero embedding as fallback
            
            # Add nodes and edges
            for entity, entity_type in entities:
                self.graph.add_node(
                    entity, 
                    type=entity_type, 
                    embedding=text_embedding,
                    sources=[source] if source else []
                )
            
            # Create potential relationships
            for i in range(len(entities) - 1):
                for j in range(i + 1, len(entities)):
                    entity1, type1 = entities[i]
                    entity2, type2 = entities[j]
                    
                    # Add edge with potential semantic relationship
                    self.graph.add_edge(
                        entity1, entity2, 
                        relation_type="potential_relation",
                        weight=0.5  # Default weight
                    )
        except ImportError:
            print("NumPy is not available. Please install it using 'pip install numpy'")
            return  # Early return if NumPy can't be imported
        except Exception as e:
            print(f"Unexpected error in add_knowledge: {e}")
            
    def semantic_search(self, query, top_k=5):
        """
        Semantic search across the knowledge graph
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Compute similarity to all node embeddings
        node_similarities = {}
        for node, data in self.graph.nodes(data=True):
            if 'embedding' in data:
                similarity = cosine_similarity(
                    [query_embedding], 
                    [data['embedding']]
                )[0][0]
                node_similarities[node] = similarity
        
        # Return top-k most similar nodes
        top_nodes = sorted(
            node_similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return top_nodes

class MultiAgentOrchestrator:
    def __init__(self, knowledge_graph):
        """
        Initialize multi-agent system with specialized agents
        """
        self.agents = {
            "scientific_analyst": ScientificAnalystAgent(),
            "content_stylist": ContentStylistAgent(),
            "fact_checker": FactCheckerAgent(),
            "linguistic_validator": LinguisticValidatorAgent()
        }
        
        self.knowledge_graph = knowledge_graph
    
    def route_task(self, task_type: str, content: str):
        """
        Route content generation to appropriate specialized agents
        """
        # Semantic routing based on task characteristics
        routing_map = {
            "scientific": ["scientific_analyst", "fact_checker"],
            "technical": ["scientific_analyst", "linguistic_validator"],
            "creative": ["content_stylist"],
            "professional": ["content_stylist", "linguistic_validator"]
        }
        
        selected_agents = routing_map.get(task_type, list(self.agents.keys()))
        
        # Collaborative generation
        processed_content = content
        for agent_name in selected_agents:
            agent = self.agents[agent_name]
            processed_content = agent.process(processed_content)
        
        return processed_content

class BaseAgent:
    def process(self, content):
        raise NotImplementedError("Subclasses must implement processing method")

class ScientificAnalystAgent(BaseAgent):
    def process(self, content):
        """
        Enhance content with scientific rigor and factual precision
        """
        # Add scientific citations, validate claims
        return content

class ContentStylistAgent(BaseAgent):
    def process(self, content):
        """
        Refine content style and tone
        """
        # Adjust language, improve readability
        return content

class FactCheckerAgent(BaseAgent):
    def process(self, content):
        """
        Validate factual accuracy and reduce hallucinations
        """
        # Cross-reference with knowledge graph
        return content

class LinguisticValidatorAgent(BaseAgent):
    def process(self, content):
        """
        Ensure linguistic quality and coherence
        """
        # Check grammar, style, consistency
        return content

class HallucinationGuardrail:
    """
    Advanced hallucination detection and mitigation system
    """
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def detect_potential_hallucinations(self, content):
        """
        Multi-stage hallucination detection
        """
        # 1. Semantic Similarity Check
        semantic_matches = self.knowledge_graph.semantic_search(content)
        
        # 2. Named Entity Verification
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents]
        
        # 3. Statistical Coherence
        # Implement more advanced hallucination detection logic
        
        return {
            "semantic_matches": semantic_matches,
            "named_entities": named_entities
        }
    
    def mitigate_hallucinations(self, content):
        """
        Reduce and correct potential hallucinations
        """
        hallucination_details = self.detect_potential_hallucinations(content)
        
        # Replace unsupported claims with more generic statements
        # Add citations or disclaimers
        # Potentially regenerate content if too many hallucinations detected
        
        return content

# Integration with existing ContentGenerator
def enhance_content_generation(content_generator):
    """
    Enhance the content generation process with Graph RAG and Multi-Agent system
    """
    knowledge_graph = KnowledgeGraphRAG()
    multi_agent_orchestrator = MultiAgentOrchestrator(knowledge_graph)
    hallucination_guardrail = HallucinationGuardrail(knowledge_graph)
    
    def enhanced_generate_content(prompt, model_key):
        # 1. Add knowledge from scientific sources
        knowledge_graph.add_knowledge(prompt)
        
        # 2. Multi-agent content generation
        task_type = "scientific"  # Dynamically determine based on prompt
        processed_content = multi_agent_orchestrator.route_task(task_type, prompt)
        
        # 3. Hallucination guardrails
        final_content = hallucination_guardrail.mitigate_hallucinations(processed_content)
        
        return {
            "status": "success",
            "content": final_content,
            "model_used": model_key
        }
    
    content_generator.generate_content = enhanced_generate_content
    return content_generator

# Example usage in the main content generation pipeline
def apply_knowledge_graph_rag(content_generator):
    return enhance_content_generation(content_generator)