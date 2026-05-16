import streamlit as st
import pandas as pd
import sys
import os

# Ensure the platform root is in the path
workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace not in sys.path:
    sys.path.insert(0, workspace)

st.set_page_config(page_title='Excel Data Agent', layout='wide', page_icon='📊')

from task1_excel_agent.agent import build_agent, run_query

# Initialize session state keys with if-not-in checks
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'agent' not in st.session_state:
    st.session_state['agent'] = None
if 'llm_provider' not in st.session_state:
    st.session_state['llm_provider'] = 'gemini'
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'uploaded_filename' not in st.session_state:
    st.session_state['uploaded_filename'] = None

# Sidebar file uploader
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Check if a new file was uploaded
    if st.session_state['uploaded_filename'] != uploaded_file.name:
        try:
            file_name = uploaded_file.name.lower()
            if file_name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_name.endswith('.xls'):
                df = pd.read_excel(uploaded_file, engine='xlrd')
            else:
                raise ValueError('Unsupported file type. Please upload an .xlsx or .xls file.')
            st.session_state['df'] = df
            st.session_state['llm_provider'] = 'gemini'
            st.session_state['agent'] = build_agent(df, provider=st.session_state['llm_provider'])
            st.session_state['history'] = []  # Clear history on new file
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.sidebar.success(f"File uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
    else:
        # File is already uploaded and processed
        df = st.session_state['df']
        if df is not None:
            st.sidebar.success(f"File uploaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")

st.title("📊 Excel Data Agent")

# Main area
if st.session_state['df'] is None:
    st.info("👋 Welcome to the Excel Data Agent! Please upload an Excel file (.xlsx or .xls) from the sidebar to get started. You can then ask questions about your data in plain English, and I will write Python Pandas code to analyze it for you.")
else:
    st.subheader("Data Preview")
    st.dataframe(st.session_state['df'].head(5), use_container_width=True)

# Render conversation history
for msg in st.session_state['history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant' and msg.get('code'):
            with st.expander('View generated code'):
                st.code(msg['code'], language='python')

# Chat interface
question = st.chat_input("Ask a question about your data...")

if question:
    if st.session_state['df'] is None or st.session_state['agent'] is None:
        st.warning('Please upload an Excel file first.')
    else:
        # Display user question
        st.session_state['history'].append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        
        # Display assistant response
        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                result = run_query(
                    st.session_state['agent'],
                    question,
                    df=st.session_state['df'],
                    provider=st.session_state['llm_provider'],
                )
                if result.get('agent') is not None:
                    st.session_state['agent'] = result['agent']
                if result.get('provider'):
                    st.session_state['llm_provider'] = result['provider']
            
            if result.get('error'):
                st.error(result['error'])
            else:
                st.markdown(result['answer'])
                if result.get('code'):
                    with st.expander('View generated code'):
                        st.code(result['code'], language='python')
                        
            # Store in history
            st.session_state['history'].append({
                'role': 'assistant',
                'content': result['answer'] if not result.get('error') else result['error'],
                'code': result.get('code', '')
            })
