# app.py
import streamlit as st
from src.models import ContentGenerator
from src.prompts import PLATFORM_TEMPLATES
from src.utils import format_prompt, post_process_content
from config import ContentRequest
import time

def main():
    st.title("AI Content Generator")
    
    # Inicializar el generador con manejo de estado
    if 'generator' not in st.session_state:
        with st.spinner("Loading model... This may take a few minutes..."):
            st.session_state.generator = ContentGenerator()
            st.success("Model loaded successfully!")
    
    # Input form
    with st.form("content_form"):
        theme = st.text_input("Theme", placeholder="e.g., AI, travel, health")
        audience = st.text_input("Target Audience", placeholder="e.g., professionals, general public")
        platform = st.selectbox("Platform", list(PLATFORM_TEMPLATES.keys()))
        context = st.text_area("Additional Context (optional)", placeholder="Any specific context or requirements")
        tone = st.selectbox("Tone", ["professional", "casual", "formal", "friendly"])
        
        submit = st.form_submit_button("Generate Content")
    
    if submit and theme and audience:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress bar
            status_text.text("Preparing prompt...")
            progress_bar.progress(25)
            
            # Create content request
            request = ContentRequest(
                theme=theme,
                audience=audience,
                platform=platform,
                context=context,
                tone=tone
            )
            
            # Get template and format prompt
            template = PLATFORM_TEMPLATES[platform]
            prompt = format_prompt(template, request)
            
            status_text.text("Generating content...")
            progress_bar.progress(50)
            
            # Generate content with timeout
            start_time = time.time()
            content = st.session_state.generator.generate_content(prompt)
            
            status_text.text("Post-processing content...")
            progress_bar.progress(75)
            
            # Post-process content
            processed_content = post_process_content(content, platform)
            
            progress_bar.progress(100)
            status_text.text("Content generated successfully!")
            
            # Display results
            st.subheader("Generated Content")
            st.text_area("Content", processed_content, height=300)
            st.download_button(
                "Download Content",
                processed_content,
                file_name=f"{platform.lower()}_content.txt"
            )
            
            # Display generation time
            generation_time = time.time() - start_time
            st.info(f"Generation time: {generation_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"Error generating content: {str(e)}")
            st.info("Please try again with different parameters or reload the page.")

if __name__ == "__main__":
    main()