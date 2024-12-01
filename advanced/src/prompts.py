# src/prompts.py
PLATFORM_TEMPLATES = {
    "Twitter/X": """Generate a 4-tweet thread about {theme} following these guidelines:

- Target Audience: {audience}
- Communication Style: {tone}
- Include Context: {context}
- Company/Personal Angle: {company_info}

Requirements:
1. Each tweet must be 280 characters or less
2. Create a coherent narrative across tweets
3. Include one unique, surprising fact
4. Use strategic hashtags
5. Ensure clear, engaging language

Output Format:
First tweet should be a compelling hook
Second tweet provides deeper insight
Third tweet adds personal or company perspective
Fourth tweet includes a clear call to action""",
    
    "LinkedIn": """Create a professional LinkedIn post about {theme}:

Target Audience: {audience}
Tone: {tone}

Post Structure:
1. Headline: Provocative, value-driven
2. Opening (50-75 words): 
   - Personal or industry narrative
   - Establish immediate relevance

3. Three Key Sections (75-100 words each):
   - Industry trend analysis
   - Strategic implementation
   - Innovative perspective

4. Incorporate: {context} and {company_info}

5. Call to Action:
   - Invite professional dialogue
   - Suggest next steps

Style Notes:
- Use professional, nuanced language
- Back claims with specific insights
- Avoid clichés and generic statements""",
    
    "Blog": """Develop a comprehensive blog post about {theme}:

Audience: {audience}
Tone: {tone}

Structure:
1. Compelling Headline
2. Introduction (150-200 words):
   - Contextual narrative
   - Clear value proposition
   - Incorporate {context}

3. Main Sections:
   a) Fundamental Overview
      - Historical context
      - Current landscape

   b) Detailed Analysis
      - Technical insights
      - Research-backed perspectives

   c) Practical Applications
      - Actionable strategies
      - Real-world implementation

4. Company/Personal Perspective:
   Integrate {company_info} to demonstrate expertise

5. Future Outlook
   - Emerging trends
   - Potential innovations

6. Conclusion:
   - Synthesize key points
   - Powerful call to action

Specifications:
- 1000-1500 words
- Clear, accessible language
- Avoid superficial content"""
}