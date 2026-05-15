# Task 1: Excel Data Agent - Developer Notes

## Agent & Tools Setup
- **Agent Type**: We are using `AgentType.OPENAI_FUNCTIONS` along with LangChain's `create_pandas_dataframe_agent`. This type forces the model to use structured function calls, making it very reliable for code execution.
- **Why `return_intermediate_steps=True` matters**: Without this flag, the agent only returns the final string answer (e.g., "The total stock is 350."). With it, the agent returns the exact Pandas code it wrote and executed to find that answer. This is vital for transparency and trust in the Excel Data Agent.
- **Code Extraction from `intermediate_steps`**: The `run_query` function safely iterates over `result['intermediate_steps']`. Each step is a tuple containing an `AgentAction`. We extract the generated Pandas code using `action.log` and join multiple steps into a single string.
- **Errors Encountered**: Upgraded LangChain dependencies required switching from `langchain.tools` to `langchain_core.tools` and handling typing for `AgentExecutor` to avoid import conflicts between core versions. Also, the agent correctly throws handled errors when `OPENROUTER_API_KEY` is not available in the environment, preventing Streamlit application crashes.

## Implementation Log
- Initialized empty project structure.
- Documented shared dependencies.
- Implemented `get_search_tool` and `get_pandas_tool` in `tools.py`.
- Implemented `build_agent` and `run_query` in `agent.py`.

## Architecture Decisions
- Modular design separated into `app.py` (UI), `agent.py` (logic), and `tools.py` (custom tools).

## Shared Dependencies
- **`shared.llm`**: Used to instantiate the Language Model via OpenRouter for agent thought processes.
- **`shared.prompts`**: Uses `EXCEL_AGENT_PREFIX` for initializing the LLM persona.
- **`shared.utils`**: Used to safely handle text inputs and JSON configurations if needed.
- **Design Choices**: The agent relies on a single source of truth for prompts and LLM instantiation, ensuring consistency and allowing easy swaps (e.g., to local models).

## Feature Notes
- Pending: Streamlit UI for Excel upload.
- Completed: Agent logic to execute Pandas code.
- Completed: Web search integration.

## Debug/Change Log
- Scaffolded files.
- Added implementation of `shared/` utilities.
- Implemented core agent execution workflow and error isolation.

## Known Limitations
- (TBD)

## Streamlit UI
- **Session State Design**: State management includes keys for `df`, `agent`, `history`, and `uploaded_filename`. 
- **Agent Rebuild Logic**: The file uploader triggers an agent rebuild (`build_agent`) only when a new file name is detected. At the same time, the `history` is cleared so previous datasets don't contaminate the chat context.
- **Demo Test Results**: The UI correctly initializes the pandas agent, previews the data via `st.dataframe`, and captures the intermediate code executions in a collapsible `st.expander` block. Tested successfully with `OPENAI_API_KEY`.
