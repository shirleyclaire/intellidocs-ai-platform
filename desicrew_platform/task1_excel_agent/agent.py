import pandas as pd
from typing import Any
from .tools import get_pandas_tool

def build_agent(df: pd.DataFrame) -> 'Any':
    """Builds and returns the pandas agent."""
    return get_pandas_tool(df)

def run_query(agent: 'Any', question: str) -> dict:
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
            'error': None
        }
    except Exception as e:
        return {
            'answer': 'I encountered an error processing that query.',
            'code': '',
            'error': str(e)
        }
