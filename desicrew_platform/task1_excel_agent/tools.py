import pandas as pd
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from typing import Any
from shared.llm import get_llm
from shared.prompts import EXCEL_AGENT_PREFIX

def get_search_tool() -> Tool:
    """Returns a LangChain Tool that wraps DuckDuckGoSearchRun."""
    search = DuckDuckGoSearchRun()
    return Tool(
        name="WebSearch",
        func=search.run,
        description="Use this to look up definitions, acronyms, or external context about data. Input: a plain English search query."
    )

def get_pandas_tool(df: pd.DataFrame) -> 'Any':
    """Returns the full LangChain Pandas dataframe agent."""
    llm = get_llm()
    
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type="openai-functions",
        verbose=True,
        return_intermediate_steps=True,
        extra_tools=[get_search_tool()],
        prefix=EXCEL_AGENT_PREFIX,
        allow_dangerous_code=True
    )
    return agent
