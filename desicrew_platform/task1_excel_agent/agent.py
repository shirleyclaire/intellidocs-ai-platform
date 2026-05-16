import pandas as pd
from typing import Any
from .tools import get_pandas_tool

def build_agent(df: pd.DataFrame, provider: str = "gemini") -> 'Any':
    """Builds and returns the pandas agent."""
    return get_pandas_tool(df, provider=provider)

def _is_quota_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return "quota" in lowered or "resource_exhausted" in lowered or "429" in lowered


def run_query(agent: 'Any', question: str, df: pd.DataFrame | None = None, provider: str = "gemini") -> dict:
    """Runs a query using the agent and safely extracts the output."""
    try:
        result = agent.invoke({'input': question})
        answer = result.get('output', '')
        
        code_blocks = []
        intermediate_steps = result.get('intermediate_steps', [])
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) > 0:
                action = step[0]
                if hasattr(action, 'log'):
                    code_blocks.append(str(action.log))
                    
        code = "\n\n".join(code_blocks)
        
        return {
            'answer': answer,
            'code': code,
            'error': None,
            'agent': agent,
            'provider': provider,
        }
    except Exception as e:
        error_text = str(e)
        if df is not None and provider == "gemini" and _is_quota_error(error_text):
            try:
                fallback_provider = "grok"
                fallback_agent = build_agent(df, provider=fallback_provider)
                fallback_result = fallback_agent.invoke({'input': question})
                answer = fallback_result.get('output', '')

                code_blocks = []
                intermediate_steps = fallback_result.get('intermediate_steps', [])
                for step in intermediate_steps:
                    if isinstance(step, tuple) and len(step) > 0:
                        action = step[0]
                        if hasattr(action, 'log'):
                            code_blocks.append(str(action.log))

                code = "\n\n".join(code_blocks)

                return {
                    'answer': answer,
                    'code': code,
                    'error': None,
                    'agent': fallback_agent,
                    'provider': fallback_provider,
                }
            except Exception as fallback_error:
                error_text = f"Gemini failed with quota error; Grok fallback also failed: {fallback_error}"

        return {
            'answer': 'I encountered an error processing that query.',
            'code': '',
            'error': error_text,
            'agent': agent,
            'provider': provider,
        }
