"""LLM utility for Desicrew AI Platform."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
import streamlit as st

def get_llm() -> ChatOpenAI:
    """
    Get a LangChain-compatible LLM instance pointing to OpenRouter.
    
    Returns:
        ChatOpenAI: Configured LLM instance.
    """
    # Fallback to Ollama:
    # To switch to Ollama locally later, replace this function body with:
    # from langchain_community.chat_models import ChatOllama
    # return ChatOllama(model="mistral", temperature=0)

    # Get API key from env, fallback to Streamlit secrets
    openrouter_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY")
    openai_key: Optional[str] = os.environ.get("OPENAI_API_KEY")

    if not openrouter_key:
        try:
            openrouter_key = st.secrets.get("OPENROUTER_API_KEY")
        except FileNotFoundError:
            pass

    if not openai_key:
        try:
            openai_key = st.secrets.get("OPENAI_API_KEY")
        except FileNotFoundError:
            pass
            
    if openrouter_key:
        return ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_key,
            model_name="mistralai/mistral-7b-instruct",
            temperature=0.0
        )
    elif openai_key:
        return ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-4o-mini",
            temperature=0.0
        )
    else:
        raise ValueError("Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set in environment or Streamlit secrets.")
