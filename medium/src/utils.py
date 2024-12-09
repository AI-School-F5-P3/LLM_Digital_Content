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
    if platform == "Twitter/X":
        # Split into tweets and add numbering
        tweets = content.split("\n\n")
        return "\n\n".join(f"Tweet {i+1}/{len(tweets)}:\n{tweet}" 
                          for i, tweet in enumerate(tweets))
        
    elif platform == "Blog":
        # Formatear como entrada de blog con párrafos
        paragraphs = content.split("\n\n")
        return "\n".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)

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
        # Devuelve el contenido sin cambios si no hay formato específico
        return content
    
def integrate_image_into_content(content: str, image_url: str, image_description: str) -> str:
    """
    Integrate an image URL and description into the generated content.
    
    Args:
        content (str): The generated text content
        image_url (str): URL of the relevant image
        image_description (str): Description of the image
    
    Returns:
        str: Content with image integration
    """
    # Split content into paragraphs
    paragraphs = content.split('\n\n')
    
    # Create Markdown image insertion
    image_insert = f"\n\n![{image_description}]({image_url} \"{image_description}\")\n\n*Image Source: {image_description}*\n\n"
    
    # If we have at least 2 paragraphs, insert the image after the first paragraph
    if len(paragraphs) >= 2:
        paragraphs.insert(1, image_insert)
        return '\n\n'.join(paragraphs)
    
    # If content is too short, append image at the end
    return content + image_insert