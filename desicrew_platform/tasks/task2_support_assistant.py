import os
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

# Switch directory so local assets resolve perfectly
script_dir = os.path.join(workspace_root, "task2_rag_assistant")
os.chdir(script_dir)

# Execute the app cleanly
runpy.run_path(os.path.join(script_dir, "app.py"), init_globals=globals())
