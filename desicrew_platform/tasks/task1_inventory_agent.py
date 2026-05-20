import os
# Force native gRPC DNS resolver to avoid hangs during imports/connections on some DNS setups (like Windows or Render)
os.environ["GRPC_DNS_RESOLVER"] = "native"
import sys
import runpy
import streamlit as st

# Setup system paths safely
curr_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(curr_dir, ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Monkeypatch st.set_page_config to avoid multi-page config crashes
st.set_page_config = lambda *args, **kwargs: None

script_dir = os.path.join(workspace_root, "task1_excel_agent")
runpy.run_path(os.path.join(script_dir, "app.py"), init_globals=globals(), run_name="__main__")
