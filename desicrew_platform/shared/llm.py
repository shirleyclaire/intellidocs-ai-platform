"""LLM utility for Desicrew AI Platform."""

import os
from typing import Optional

import streamlit as st

# Force native gRPC DNS resolver to avoid hangs during imports/connections on some DNS setups (like Windows or Render)
os.environ["GRPC_DNS_RESOLVER"] = "native"


def _secret_or_env(*names: str) -> Optional[str]:
    # 1. Check environment variables first
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()

    # 2. Check Streamlit secrets
    try:
        # Check direct/flat secrets (e.g. st.secrets["GEMINI_API_KEY"])
        for name in names:
            value = st.secrets.get(name)
            if value and isinstance(value, str):
                return value.strip()
            
        # Check nested structures (e.g. st.secrets["gemini"]["api_key"])
        for name in names:
            name_lower = name.lower()
            if "gemini" in name_lower:
                val = st.secrets.get("gemini", {}).get("api_key")
                if val:
                    return val.strip()
            elif "openai" in name_lower:
                val = st.secrets.get("openai", {}).get("api_key")
                if val:
                    return val.strip()
            elif "grok" in name_lower or "xai" in name_lower:
                val = st.secrets.get("grok", {}).get("api_key") or st.secrets.get("xai", {}).get("api_key")
                if val:
                    return val.strip()
    except Exception:
        pass

    return None


def get_llm(provider: str = "gemini"):
    """Return a chat model for the requested provider."""
    provider = provider.lower()

    if provider == "gemini":
        gemini_key = _secret_or_env("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in environment or Streamlit secrets. "
                "If you are deployed on Streamlit Cloud, please go to your App Settings -> Secrets console, "
                "and paste your TOML secrets there:\n\n"
                "[gemini]\napi_key = \"YOUR_KEY_HERE\""
            )

        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=gemini_key,
            model="gemini-2.5-flash",
            temperature=0.0,
        )

    if provider == "grok":
        grok_key = _secret_or_env("GROK_API_KEY", "XAI_API_KEY")
        if not grok_key:
            raise ValueError(
                "GROK_API_KEY or XAI_API_KEY is not set in environment or Streamlit secrets. "
                "If you are deployed on Streamlit Cloud, please go to your App Settings -> Secrets console, "
                "and paste your TOML secrets there:\n\n"
                "[grok]\napi_key = \"YOUR_KEY_HERE\""
            )

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            openai_api_base="https://api.x.ai/v1",
            openai_api_key=grok_key,
            model_name="grok-4.3",
            temperature=0.0,
        )

    raise ValueError(f"Unsupported provider: {provider}")
