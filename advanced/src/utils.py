# src/utils.py
def format_prompt(template: str, request: 'ContentRequest') -> str:
    return template.format(
        theme=request.theme,
        audience=request.audience,
        tone=request.tone,
        context=request.context,
        company_info=request.company_info or ""  # Add this line to handle company_info
    )

def post_process_content(content: str, platform: str) -> str:
    # Remove any markdown or HTML-like formatting
    def clean_content(text):
        # Remove markdown headers
        text = text.replace('#', '').strip()
        # Remove HTML tags
        text = text.replace('<p>', '').replace('</p>', '').strip()
        return text

    # Platform-specific post-processing
    if platform == "Twitter/X":
        # Split into tweets and clean
        tweets = content.split('\n')
        cleaned_tweets = [clean_content(tweet) for tweet in tweets if tweet.strip()]
        return '\n\n'.join(cleaned_tweets)
        
    elif platform == "Blog":
        # Clean and join paragraphs
        paragraphs = content.split('\n')
        cleaned_paragraphs = [clean_content(para) for para in paragraphs if para.strip()]
        return '\n\n'.join(cleaned_paragraphs)
    
    elif platform == "LinkedIn":
        # Clean and format for professional look
        paragraphs = content.split('\n')
        cleaned_paragraphs = [clean_content(para) for para in paragraphs if para.strip()]
        return '\n\n'.join(cleaned_paragraphs)

    elif platform == "Instagram":
        # Adaptar a un estilo atractivo con hashtags y línea separadora
        hashtags = "#content #creativity #platform_specific"
        return f"{content}\n\n---\n{hashtags}"

    elif platform == "Divulgación":
        # Simplificar el lenguaje para una audiencia general
        simple_content = content.replace(",", ".").replace("however", "but").replace("therefore", "so")
        return f"Did you know? {simple_content}"

    elif platform == "Infantil":
        # Usar un lenguaje amigable y estructurado con viñetas
        sentences = content.split(". ")
        bullets = "\n".join(f"• {sentence.strip()}." for sentence in sentences if sentence.strip())
        return f"Hello, little ones! Let's learn something cool today:\n\n{bullets}"

    else:
        # Default: just clean the content
        return clean_content(content)