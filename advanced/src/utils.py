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
    def clean_content(text):
        """Remove any markdown or HTML-like formatting."""
        # Remove markdown headers
        text = text.replace('#', '').strip()
        # Remove HTML tags
        text = text.replace('<p>', '').replace('</p>', '').strip()
        return text

    def add_formatting(text, platform):
        """Add platform-specific formatting."""
        if platform == "Blog":
            # Add headers and paragraphs
            lines = text.split('\n')
            formatted_lines = [f"## {line}" if i % 3 == 0 else line for i, line in enumerate(lines)]
            return '\n'.join(formatted_lines)
        
        if platform == "LinkedIn":
            # Bold only specific sections or headings
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                if line.strip().startswith(('Título:', 'Análisis', 'Implementación', 'Perspectiva', 'Llamado')):
                    formatted_lines.append(f"**{line}**")
                else:
                    formatted_lines.append(line)
                    
            # Add relevant hashtags
            hashtags = "\n\n#ProfessionalDevelopment #IndustryInsights #CareerGrowth #BusinessStrategy"
            return '\n'.join(formatted_lines) + hashtags
        
        return text

    # Clean the content
    cleaned_content = clean_content(content)
    
    # Platform-specific processing and formatting
    if platform == "Twitter/X":
        # Split into tweets and clean
        tweets = cleaned_content.split('\n')
        cleaned_tweets = [clean_content(tweet) for tweet in tweets if tweet.strip()]
        return '\n\n'.join(cleaned_tweets)
        
    elif platform == "Blog":
        # Clean, format, and join paragraphs
        paragraphs = cleaned_content.split('\n')
        cleaned_paragraphs = [clean_content(para) for para in paragraphs if para.strip()]
        formatted_content = add_formatting('\n'.join(cleaned_paragraphs), platform)
        return formatted_content
    
    elif platform == "LinkedIn":
        # Clean, format, and structure for professional look
        paragraphs = cleaned_content.split('\n')
        cleaned_paragraphs = [clean_content(para) for para in paragraphs if para.strip()]
        formatted_content = add_formatting('\n'.join(cleaned_paragraphs), platform)
        return formatted_content

    elif platform == "Instagram":
        # Adapt to an attractive style with hashtags and separator line
        hashtags = "#content #creativity #platform_specific"
        return f"{cleaned_content}\n\n---\n{hashtags}"

    else:
        # Default: just clean the content
        return cleaned_content
