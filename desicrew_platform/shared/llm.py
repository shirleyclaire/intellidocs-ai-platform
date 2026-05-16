"""LLM utility for Desicrew AI Platform."""

import os
from typing import Optional

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def _secret_or_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value

    try:
        for name in names:
            value = st.secrets.get(name)
            if value:
                return value
    except FileNotFoundError:
        pass

    return None


def get_llm(provider: str = "gemini"):
    """Return a chat model for the requested provider."""
    provider = provider.lower()

    if provider == "gemini":
        gemini_key = _secret_or_env("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or Streamlit secrets.")

        return ChatGoogleGenerativeAI(
            google_api_key=gemini_key,
            model="gemini-2.5-flash",
            temperature=0.0,
        )

    if provider == "grok":
        grok_key = _secret_or_env("GROK_API_KEY", "XAI_API_KEY")
        if not grok_key:
            raise ValueError("GROK_API_KEY or XAI_API_KEY is not set in environment or Streamlit secrets.")

        return ChatOpenAI(
            openai_api_base="https://api.x.ai/v1",
            openai_api_key=grok_key,
            model_name="grok-4.3",
            temperature=0.0,
        )

    raise ValueError(f"Unsupported provider: {provider}")
