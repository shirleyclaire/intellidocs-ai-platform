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


def get_llm():
    """Get a chat model using Gemini first, then Grok/xAI as fallback."""
    gemini_key = _secret_or_env("GEMINI_API_KEY")
    grok_key = _secret_or_env("GROK_API_KEY", "XAI_API_KEY")

    if gemini_key:
        for gemini_model in ("gemini-2.5-flash", "gemini-2.0-flash"):
            try:
                return ChatGoogleGenerativeAI(
                    google_api_key=gemini_key,
                    model=gemini_model,
                    temperature=0.0,
                )
            except Exception:
                continue

    if grok_key:
        for grok_model in ("grok-2-latest", "grok-3-latest"):
            try:
                return ChatOpenAI(
                    openai_api_base="https://api.x.ai/v1",
                    openai_api_key=grok_key,
                    model_name=grok_model,
                    temperature=0.0,
                )
            except Exception:
                continue

    raise ValueError("Neither GEMINI_API_KEY nor GROK_API_KEY/XAI_API_KEY is set in environment or Streamlit secrets.")
