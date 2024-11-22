# src/utils.py
def format_prompt(template: str, request: 'ContentRequest') -> str:
    return template.format(
        theme=request.theme,
        audience=request.audience,
        tone=request.tone,
        context=request.context
    )

def post_process_content(content: str, platform: str) -> str:
    if platform == "Twitter/X":
        # Split into tweets and add numbering
        tweets = content.split("\n\n")
        return "\n\n".join(f"Tweet {i+1}/{len(tweets)}:\n{tweet}" 
                          for i, tweet in enumerate(tweets))
    return content