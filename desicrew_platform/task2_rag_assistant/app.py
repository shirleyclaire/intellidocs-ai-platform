import streamlit as st
import os
import shutil
from task2_rag_assistant.ingest import ingest_documents
from task2_rag_assistant.memory import build_memory, is_topic_switch
from task2_rag_assistant.retriever import build_chain

# Page config
st.set_page_config(page_title="Document Support Assistant", layout="wide", page_icon="📄")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "memory" not in st.session_state:
    st.session_state.memory = build_memory()
if "chain" not in st.session_state:
    st.session_state.chain = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = "./chroma_db"

st.title("📄 Document Support Assistant")

# Sidebar
st.sidebar.header("Knowledge Base Setup")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents", 
    type=["pdf", "txt", "docx"], 
    accept_multiple_files=True
)

if st.sidebar.button("Build Knowledge Base"):
    if uploaded_files:
        # Create docs directory
        docs_dir = "./task2_docs"
        if os.path.exists(docs_dir):
            shutil.rmtree(docs_dir)
        os.makedirs(docs_dir)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        with st.spinner("Ingesting documents..."):
            try:
                # Ingest documents
                store = ingest_documents(file_paths, persist_dir=st.session_state.persist_dir)
                
                # Rebuild chain
                st.session_state.chain = build_chain(st.session_state.memory, persist_dir=st.session_state.persist_dir)
                
                st.sidebar.success(f"Knowledge base built! {len(file_paths)} documents indexed.")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    else:
        st.sidebar.warning("Please upload files first.")

# Main chat area
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(source)

# User input
if prompt := st.chat_input("Ask a question about your documents"):
    if st.session_state.chain is None:
        st.warning("Please upload documents and build the knowledge base first.")
    else:
        # Check for topic switch
        if is_topic_switch(st.session_state.last_question, prompt):
            st.session_state.memory.clear()
            st.info("Topic switched — starting fresh context.")
            # We don't need to rebuild the chain, just clearing memory is enough as ConversationalRetrievalChain uses the memory object
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.history.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({"question": prompt})
                answer = response["answer"]
                source_docs = response.get("source_documents", [])
                
                # Format sources
                formatted_sources = []
                for doc in source_docs:
                    source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", "unknown")
                    formatted_sources.append(f"📄 {source_name} — Page {page}")
                
                st.markdown(answer)
                if formatted_sources:
                    with st.expander("Sources"):
                        for source in formatted_sources:
                            st.write(source)
                
                # Store in history
                st.session_state.history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": formatted_sources
                })
                
                st.session_state.last_question = prompt
