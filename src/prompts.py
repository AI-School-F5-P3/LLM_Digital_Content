# src/prompts.py
PLATFORM_TEMPLATES = {
    "Twitter/X": """Create a engaging tweet thread (max 280 chars per tweet) about {theme}.
    Target audience: {audience}
    Tone: {tone}
    Context: {context}
    Guidelines:
    - Make it concise and impactful
    - Use appropriate hashtags
    - Create 3-5 connected tweets
    - Encourage engagement
    """,
    
    "LinkedIn": """Create a professional LinkedIn post about {theme}.
    Target audience: {audience}
    Tone: {tone}
    Context: {context}
    Guidelines:
    - Start with a hook
    - Include professional insights
    - Add 3-5 relevant hashtags
    - End with a call to action
    - Keep it under 3000 characters
    """,
    
    "Blog": """Write a blog post about {theme}.
    Target audience: {audience}
    Tone: {tone}
    Context: {context}
    Guidelines:
    - Create an engaging headline
    - Include an introduction
    - Break into 3-4 main sections
    - Add a conclusion
    - Use subheadings
    - Include transition sentences
    """,
    
    "Instagram": """Create an Instagram caption about {theme}.
    Target audience: {audience}
    Tone: {tone}
    Context: {context}
    Guidelines:
    - Keep it engaging and concise
    - Use appropriate emojis
    - Add relevant hashtags
    - Include a call to action
    - Max 2200 characters
    """
}