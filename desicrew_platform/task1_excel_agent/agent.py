import ast
import re

import pandas as pd
from typing import Any
from .tools import get_pandas_tool


def _clean_answer_text(answer: str) -> str:
    lines = [line.strip() for line in str(answer).splitlines() if line.strip()]
    if not lines:
        return ""

    cleaned_lines = []
    for line in lines:
        line = re.sub(r"\s+", " ", line)
        line = line.replace("∗", "-").replace("*", "-")
        cleaned_lines.append(line)

    return "\n\n".join(cleaned_lines)


def _extract_code_from_steps(intermediate_steps: list[Any]) -> str:
    code_blocks = []

    for step in intermediate_steps:
        if not isinstance(step, tuple) or len(step) == 0:
            continue

        action = step[0]
        tool_name = getattr(action, "tool", "")

        if tool_name == "python_repl_ast":
            tool_input = getattr(action, "tool_input", "")
            if tool_input:
                code_blocks.append(str(tool_input).strip())
                continue

        log_text = getattr(action, "log", "")
        if log_text:
            match = re.search(r"Invoking:\s*`python_repl_ast`\s*with\s*`\{?'query':\s*'(.+?)'\}?`", str(log_text), re.DOTALL)
            if match:
                extracted = match.group(1)
                try:
                    code_blocks.append(ast.literal_eval(f"'{extracted}'"))
                except Exception:
                    code_blocks.append(extracted)
            else:
                code_blocks.append(str(log_text).strip())

    return "\n\n".join(code_blocks)

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
        answer = _clean_answer_text(result.get('output', ''))
        code = _extract_code_from_steps(result.get('intermediate_steps', []))
        
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
                answer = _clean_answer_text(fallback_result.get('output', ''))
                code = _extract_code_from_steps(fallback_result.get('intermediate_steps', []))

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
