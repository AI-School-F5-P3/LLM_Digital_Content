# src/prompts.py
PLATFORM_TEMPLATES = {
    "Twitter/X": """You are an expert social media content creator specializing in crafting engaging, concise tweet threads. 

TASK: Create a 4-tweet thread about {theme} that is informative, engaging, and tailored to {audience}.

STRICT GUIDELINES:
- Each tweet must be exactly 280 characters or less
- Use a {tone} communication style
- Include at least one unique, surprising fact
- Incorporate storytelling elements
- Use strategic hashtags
- Ensure coherent narrative across all tweets

REQUIRED CONTENT STRUCTURE:
Tweet 1 (Hook): 
- Grab attention immediately
- Pose a provocative question or share a startling statistic
- Directly relate to {theme}

Tweet 2 (Context):
- Provide deeper insight
- Use {context} to add credibility
- Include a specific, actionable insight

Tweet 3 (Personal/Company Angle):
- Integrate {company_info}
- Share a unique perspective or personal experience
- Connect the theme to broader implications

Tweet 4 (Call to Action):
- Encourage engagement
- Ask a thought-provoking question
- Provide a clear next step for the audience

EXAMPLE FORMAT:
🔥 Tweet 1/4: [Attention-grabbing hook]
🧠 Tweet 2/4: [Deeper context]
💡 Tweet 3/4: [Personal insight]
🚀 Tweet 4/4: [Call to action]

Hashtag Strategy: 
- 2-3 relevant hashtags
- Mix of broad and specific tags

ABSOLUTELY AVOID:
- Generic statements
- Jargon
- Disconnected facts
- Lack of narrative flow
""",
    
    "LinkedIn": """You are a top-tier content strategist creating a LinkedIn post that delivers maximum professional value.

STRATEGIC CONTENT GENERATION FRAMEWORK:

OBJECTIVE: Craft a sophisticated, insight-driven LinkedIn post about {theme}

CORE REQUIREMENTS:
- Length: 250-400 words
- Audience: {audience}
- Tone: {tone}
- Integration: {company_info} and {context}

PRECISE CONTENT ARCHITECTURE:

1. HEADLINE STRATEGY:
- Provocative, metrics-driven headline
- Immediately communicate unique value proposition
- Spark intellectual curiosity

2. NARRATIVE FRAMEWORK:
a) Opening Paragraph (50-75 words)
- Personal or industry-level narrative
- Establish immediate relevance
- Create emotional/professional connection

b) Insight Blocks (3 strategic sections, 75-100 words each)
- Section 1: Industry Trend Analysis
  * Quantitative data point
  * Comparative insight
  * Future projection

- Section 2: Strategic Implementation
  * Actionable framework
  * Potential challenges
  * Mitigation strategies

- Section 3: Transformative Perspective
  * Innovative approach
  * Counterintuitive observation
  * Potential breakthrough

3. COMPANY/PERSONAL CONTEXT:
- Seamless integration of {company_info}
- Demonstrate thought leadership
- Provide credibility through experience

4. CALL TO ACTION:
- Explicit engagement prompt
- Invite professional dialogue
- Suggest next learning steps

5. PROFESSIONAL SIGNALING:
- Carefully selected hashtags
- Industry-specific terminology
- Nuanced language

TONE CALIBRATION:
- {tone} communication style
- Balance between authoritative and approachable
- Use language that resonates with {audience}

FORBIDDEN ELEMENTS:
- Clichés
- Unsupported claims
- Lack of specificity
- Generic motivational statements
""",
    
    "Blog": """COMPREHENSIVE CONTENT GENERATION PROTOCOL

OBJECTIVE: Produce a meticulously structured, deeply researched blog post about {theme}

STRATEGIC COMPONENTS:

1. HEADLINE ENGINEERING:
- Provocative, SEO-optimized headline
- Promise of unique value
- Instantaneous reader engagement

2. INTRODUCTION (150-200 words):
- Contextual narrative
- Personal/industry anecdote
- Clear value proposition
- Incorporate {context}

3. CONTENT ARCHITECTURE:
a) Section 1: Fundamental Landscape
- Comprehensive overview
- Historical context
- Current state of {theme}

b) Section 2: Deep Analysis
- Technical insights
- Research-backed perspectives
- Cutting-edge developments

c) Section 3: Practical Application
- Actionable strategies
- Real-world implementation
- Potential challenges and solutions

4. COMPANY/PERSONAL INTEGRATION:
- Strategic placement of {company_info}
- Demonstrate expertise
- Create credibility bridge

5. ADVANCED INSIGHTS SECTION:
- Forward-looking predictions
- Emerging trends
- Potential disruptions

6. CONCLUSION:
- Synthesize key points
- Future outlook
- Powerful call to action

TONE CALIBRATION:
- {tone} communication style
- Tailored to {audience}
- Balance between academic rigor and accessibility

TECHNICAL REQUIREMENTS:
- 1000-1500 words
- Markdown compatible
- SEO-friendly structure

PROHIBITED ELEMENTS:
- Superficial content
- Unsupported claims
- Lack of original perspective
"""
}