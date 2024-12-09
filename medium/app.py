import streamlit as st
from src.models import ContentGenerator
from src.prompts import PLATFORM_TEMPLATES
from src.utils import format_prompt, post_process_content, integrate_image_into_content
from src.image_utils import ImageRetriever  # New import
from config import ContentRequest, ModelConfig
import time

def main():
    st.title("AI Content Generator - Enhanced Version")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Loading initial model... This may take a few minutes..."):
            st.session_state.generator = ContentGenerator()
            st.success("System initialized successfully!")
    
    # Initialize image retriever
    if 'image_retriever' not in st.session_state:
        st.session_state.image_retriever = ImageRetriever()
    
    # Sidebar for model selection and advanced options
    with st.sidebar:
        st.header("Advanced Options")
        
        # Get available models, handling potential access issues
        available_models = list(ModelConfig.AVAILABLE_MODELS.keys())
        
        if not available_models:
            st.error("No models are currently available. Check your Hugging Face access.")
            selected_model = None
        else:
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                format_func=lambda x: f"{x.title()} - {ModelConfig.AVAILABLE_MODELS[x]['description']}"
            )
        
        include_image = st.checkbox("Include AI-generated image")
        
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
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepare request
            status_text.text("Preparing content generation...")
            progress_bar.progress(25)
            
            # Image retrieval
            retrieved_images = []
            if include_image:
                status_text.text("Retrieving relevant images...")
                retrieved_images = st.session_state.image_retriever.get_relevant_image(theme)
                if retrieved_images:
                    st.image(retrieved_images[0]['url'], caption=retrieved_images[0]['description'])
            
            request = ContentRequest(
                theme=theme,
                audience=audience,
                platform=platform,
                context=context,
                tone=tone,
                company_info=company_info,
                selected_model=selected_model,
                include_image=include_image
            )
            
            # Generate content
            status_text.text(f"Generating content using {selected_model.title()} model...")
            progress_bar.progress(50)
            
            start_time = time.time()
            
            # Get and format prompt
            template = PLATFORM_TEMPLATES[platform]
            prompt = format_prompt(template, request)
            
            # Add image context to prompt if image retrieved
            if retrieved_images:
                prompt += f"\n\nRelevant Image URL: {retrieved_images[0]['url']}"
                prompt += f"\nImage Description: {retrieved_images[0]['description']}"
            
            # Generate content
            result = st.session_state.generator.generate_content(prompt, selected_model)
            
            if result["status"] == "success"
                # More stringent content validation
                if len(result["content"].split()) < 50:
                    st.warning("Generated content is too brief. Regenerating...")
                    # Potentially retry generation or show error
                else:
                    # Comment this out integrate image into content if available
                    # if include_image and retrieved_images:
                    #     result["content"] = integrate_image_into_content(
                    #         result["content"], 
                    #         retrieved_images[0]['url'], 
                    #         retrieved_images[0]['description']
                    #     )
                    processed_content = post_process_content(result["content"], platform)
                # Post-process
                status_text.text("Post-processing content...")
                progress_bar.progress(75)
                
                processed_content = post_process_content(result["content"], platform)
                
                # Display results
                progress_bar.progress(100)
                status_text.text("Content generated successfully!")
                
                st.subheader("Generated Content")
                st.text_area("Content", processed_content, height=300) # Change text_area for "markdown" to render markdown and delete the height
                
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
            st.error(f"Error: {str(e)}")
            st.info("Please try again with different parameters or reload the page.")

if __name__ == "__main__":
    main()