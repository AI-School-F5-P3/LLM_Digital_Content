# app.py
import streamlit as st
from src.models import ContentGenerator
from src.prompts import PLATFORM_TEMPLATES
from src.utils import format_prompt, post_process_content
from src.image_utils import ImageRetriever
from config import ContentRequest, ModelConfig
import time
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def main():
    st.title("AI Content Generator - Advanced Version")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Loading initial model... This may take a few minutes..."):
            st.session_state.generator = ContentGenerator()
            st.success("System initialized successfully!")
    
    # Initialize image retriever
    if 'image_retriever' not in st.session_state:
        st.session_state.image_retriever = ImageRetriever()
    
    # Sidebar for advanced options
    with st.sidebar:
        st.header("Advanced Options")
        
        # Model Selection
        available_models = list(ModelConfig.AVAILABLE_MODELS.keys())
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            format_func=lambda x: f"{x.title()} - {ModelConfig.AVAILABLE_MODELS[x]['description']}"
        )
        
        # Language Selection
        selected_language = st.selectbox(
            "Content Language",
            list(ModelConfig.SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: ModelConfig.SUPPORTED_LANGUAGES[x]
        )
        
        # Image repository with Pixabay as default
        image_repository = st.selectbox(
            "Image Repository",
            ["Pixabay", "Unsplash", "AI Image Generation"],
            index=0  # Set Pixabay as default
        )
        
        scientific_mode = st.checkbox("Scientific Content Mode")
        financial_news_mode = st.checkbox("Financial News Mode")
    
    # Main form
    with st.form("content_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.text_input("Theme", placeholder="e.g., AI, travel, health")
            audience = st.text_input("Target Audience", placeholder="e.g., professionals, general public")
            platform = st.selectbox("Platform", list(PLATFORM_TEMPLATES.keys()))
        
        with col2:
            context = st.text_area("Additional Context", placeholder="Any specific context or requirements")
            company_info = st.text_area("Company/Personal Info", placeholder="Add personalization details")
            tone = st.selectbox("Tone", ["professional", "casual", "formal", "friendly"])
        
        submit = st.form_submit_button("Generate Content")
    
    if submit and theme and audience:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Scientific Mode Processing
            if scientific_mode:
                scientific_docs = st.session_state.generator.generate_scientific_content(theme, selected_language)
                if scientific_docs.get('status') == 'error':
                    st.warning(f"Scientific content retrieval failed: {scientific_docs.get('message', 'Unknown error')}")
                else:
                    context += f"\n\nScientific Context: {scientific_docs}"
            
            # Financial News Mode
            if financial_news_mode:
                status_text.text("Fetching latest financial market information...")
                financial_news = st.session_state.generator.generate_financial_news(selected_language)
                context += f"\n\nFinancial News: {financial_news}"
            
            # Image Retrieval
            retrieved_images = []
            status_text.text("Retrieving relevant images...")
            try:
                if image_repository == "AI Image Generation":
                    retrieved_images = st.session_state.image_retriever.generate_ai_image(theme)
                elif image_repository == "Unsplash":
                    retrieved_images = st.session_state.image_retriever.get_relevant_image(theme)
                else:  # Pixabay
                    clean_theme = ''.join(
                        char for char in theme.replace('\n', ' ').strip() 
                        if char.isalnum() or char.isspace()
                    )[:50]
                    retrieved_images = st.session_state.image_retriever.get_pixabay_images(clean_theme)

                # Image display logic
                if retrieved_images:
                    image_url = retrieved_images[0].get('url')
                    image_description = retrieved_images[0].get('description', theme)
                    
                    st.image(
                        image_url, 
                        caption=image_description,
                        use_container_width=True
                    )
                else:
                    st.warning(f"No images found for theme: {theme}")

            except Exception as e:
                st.error(f"Image retrieval error: {e}")
            
            # Truncate context if it's too long
            if len(context) > 500:
                context = context[:500]
            
            request = ContentRequest(
                theme=theme[:50],  # Ensure theme is not too long
                audience=audience[:50],
                platform=platform,
                context=context,
                tone=tone,
                company_info=company_info[:300] if company_info else None,
                selected_model=selected_model,
                language=selected_language
            )
            
            # Generate content
            status_text.text(f"Generating content using {selected_model.title()} model...")
            progress_bar.progress(50)
            
            start_time = time.time()
            
            # Get and format prompt
            template = PLATFORM_TEMPLATES[platform]
            prompt = format_prompt(template, request)
            prompt += f"\nLanguage: {selected_language}"
            prompt += f"\nStrict Instructions: Generate content exclusively in {ModelConfig.SUPPORTED_LANGUAGES[selected_language]}"
            
            # Generate content
            result = st.session_state.generator.generate_content(prompt, selected_model)
            
            if result["status"] == "success":
                # More stringent content validation
                if len(result["content"].split()) < 50:
                    st.warning("Generated content is too brief. Regenerating...")
                    # Potentially retry generation or show error
                else:
                    processed_content = post_process_content(result["content"], platform)
                # Post-process
                status_text.text("Post-processing content...")
                progress_bar.progress(75)
                
                processed_content = post_process_content(result["content"], platform)
                
                # Display results
                progress_bar.progress(100)
                status_text.text("Content generated successfully!")
                
                st.subheader("Generated Content")
                st.text_area("Content", processed_content, height=300)
                
                # Metadata and downloads
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Content",
                        processed_content,
                        file_name=f"{platform.lower()}_content.txt"
                    )
                with col2:
                    generation_time = time.time() - start_time
                    st.info(f"Generation time: {generation_time:.2f} seconds")
                    st.info(f"Model used: {result['model_used']}")
            else:
                st.error(result["content"])
                st.info("Unable to generate content. Please adjust parameters and try again.")
                
        except Exception as e:
            st.error(f"Comprehensive error: {str(e)}")
            # Log the full traceback for debugging
            import traceback
            st.text(traceback.format_exc())

if __name__ == "__main__":
    main()