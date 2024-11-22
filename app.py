# app.py
import streamlit as st
from src.models import ContentGenerator
from src.prompts import PLATFORM_TEMPLATES
from src.utils import format_prompt, post_process_content
from config import ContentRequest

def main():
    st.title("AI Content Generator")
    
    # Initialize content generator
    if 'generator' not in st.session_state:
        st.session_state.generator = ContentGenerator()
    
    # Input form
    with st.form("content_form"):
        theme = st.text_input("Theme", placeholder="e.g., AI, travel, health")
        audience = st.text_input("Target Audience", placeholder="e.g., professionals, general public")
        platform = st.selectbox("Platform", list(PLATFORM_TEMPLATES.keys()))
        context = st.text_area("Additional Context (optional)", placeholder="Any specific context or requirements")
        tone = st.selectbox("Tone", ["professional", "casual", "formal", "friendly"])
        
        submit = st.form_submit_button("Generate Content")
    
    if submit and theme and audience:
        with st.spinner("Generating content..."):
            # Create content request
            request = ContentRequest(
                theme=theme,
                audience=audience,
                platform=platform,
                context=context,
                tone=tone
            )
            
            # Get platform-specific template
            template = PLATFORM_TEMPLATES[platform]
            
            # Format prompt
            prompt = format_prompt(template, request)
            
            # Generate content
            content = st.session_state.generator.generate_content(prompt)
            
            # Post-process content
            processed_content = post_process_content(content, platform)
            
            # Display results
            st.subheader("Generated Content")
            st.text_area("Content", processed_content, height=300)
            st.download_button(
                "Download Content",
                processed_content,
                file_name=f"{platform.lower()}_content.txt"
            )

if __name__ == "__main__":
    main()