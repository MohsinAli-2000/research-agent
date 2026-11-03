import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url

load_dotenv()

# Standardize Gemini API key across environment variables
def get_gemini_api_key():
    key = (
        os.getenv("GOOGLE_GEMINI_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        # Remove redundant duplicate key to avoid Google SDK warning
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]
    return key


def extract_text_content(content) -> str:
    """Extract clean string text from string, list of blocks, or AIMessage content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text"))
        return "".join(parts)
    return str(content) if content is not None else ""


# Model Initialization helper with automatic quota failover
def get_llm(model: str = "gemini-3-flash-preview", temperature: float = 0.0, api_key: str = None):
    key = api_key or get_gemini_api_key()
    
    primary = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=key if key else None,
        max_retries=2,
    )
    
    # Candidate fallback models to ensure high availability and quota resilience
    candidates = [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
    ]
    fallback_models = [m for m in candidates if m != model]
    
    fallbacks = [
        ChatGoogleGenerativeAI(
            model=fb,
            temperature=temperature,
            google_api_key=key if key else None,
            max_retries=1,
        )
        for fb in fallback_models[:2]
    ]
    
    return primary.with_fallbacks(fallbacks)

llm = get_llm()

# 1st Agent : Search Agent
def build_search_agent(custom_llm=None):
    return create_agent(
        model=custom_llm or llm,
        tools=[web_search],
    )

# 2nd Agent : Reader Agent
def build_reader_agent(custom_llm=None):
    return create_agent(
        model=custom_llm or llm,
        tools=[scrape_url],
    )

# Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

def get_writer_chain(custom_llm=None):
    return writer_prompt | (custom_llm or llm) | StrOutputParser()

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

def get_critic_chain(custom_llm=None):
    return critic_prompt | (custom_llm or llm) | StrOutputParser()

critic_chain = critic_prompt | llm | StrOutputParser()
