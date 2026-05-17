import streamlit as st
import os
import shutil
import sys

# Ensure the platform root is on the import path when running this file with Streamlit.
workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace not in sys.path:
    sys.path.insert(0, workspace)

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
        # Reset st.session_state.chain and force garbage collection to release active SQLite locks on Windows
        st.session_state.chain = None
        import gc
        gc.collect()
        
        # Clear out the persistence directory (Chroma DB) to prevent SQLite locks and document duplicates
        persist_dir = st.session_state.persist_dir
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
            except Exception as e:
                print(f"Chroma connection clean warning: {e}")
                
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
        if msg.get("generated_question"):
            st.caption(f"🧠 *Used conversation history to clarify context: \"{msg['generated_question']}\"*")
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(source)

# User input
if prompt := st.chat_input("Ask a question about your documents"):
    if st.session_state.chain is None:
        st.warning("Please upload documents and build the knowledge base first.")
    else:
        # Check for topic switch using the optimal 0.22 similarity threshold
        if is_topic_switch(st.session_state.last_question, prompt, threshold=0.22):
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
                generated_question = response.get("generated_question", "").strip()
                
                # Deduplicate & convert page indexes to 1-indexed (guard against 0-index)
                unique_sources = []
                seen_sources = set()
                for doc in source_docs:
                    source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page = doc.metadata.get("page", 0)
                    
                    try:
                        # Mathematically shift 0-indexed database page numbers to 1-indexed UI page numbers
                        page_val = int(page) + 1
                    except (ValueError, TypeError):
                        page_val = page
                        
                    pair = (source_name, page_val)
                    if pair not in seen_sources:
                        seen_sources.add(pair)
                        unique_sources.append(pair)
                
                # Smart Citation Filtering (Requirement 3)
                # Only keep sources whose names are explicitly mentioned in the LLM's response
                # to prevent showing unrelated documents that the vector retriever pulled in.
                answer_lower = answer.lower()
                has_explicit_mention = False
                for source_name, _ in unique_sources:
                    # Clean filename e.g. "general_policy.pdf" -> "general_policy" or "general policy"
                    base_clean = os.path.splitext(source_name)[0].lower()
                    if base_clean in answer_lower or base_clean.replace('_', ' ') in answer_lower:
                        has_explicit_mention = True
                        break
                
                filtered_sources = []
                if has_explicit_mention:
                    for source_name, page_val in unique_sources:
                        base_clean = os.path.splitext(source_name)[0].lower()
                        if base_clean in answer_lower or base_clean.replace('_', ' ') in answer_lower:
                            filtered_sources.append((source_name, page_val))
                else:
                    # Fallback to all unique sources if no document filenames are directly cited
                    filtered_sources = unique_sources
                
                # Format final sources for display
                formatted_sources = [f"📄 {name} — Page {page}" for name, page in filtered_sources]
                
                # Render reconstructed query if it's different from the original prompt (Requirement 2)
                # And only if there actually is past context/history to draw from!
                rewritten = None
                if len(st.session_state.history) > 1 and generated_question:
                    orig_clean = prompt.strip().lower()
                    gen_clean = generated_question.strip().lower()
                    if orig_clean != gen_clean:
                        rewritten = generated_question
                        st.caption(f"🧠 *Used conversation history to clarify context: \"{rewritten}\"*")
                
                st.markdown(answer)
                if formatted_sources:
                    with st.expander("Sources"):
                        for source in formatted_sources:
                            st.write(source)
                
                # Store in history
                st.session_state.history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": formatted_sources,
                    "generated_question": rewritten
                })
                
                st.session_state.last_question = prompt
