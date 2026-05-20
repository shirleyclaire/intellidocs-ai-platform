import os
# Force native gRPC DNS resolver to avoid hangs during imports/connections on some DNS setups (like Windows or Render)
os.environ["GRPC_DNS_RESOLVER"] = "native"
# Configure OpenMP environment variables to prevent PaddleOCR crashes and limit core threading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st

# Must be called immediately at the top of the hub script before any other rendering
st.set_page_config(page_title="Enterprise AI & IDP Suite", page_icon="💼", layout="wide")

def show_landing_hub():
    """
    Renders the beautifully formatted Enterprise Suite Landing Hub.
    """
    st.title("💼 Enterprise AI & Intelligent Document Processing Suite")
    st.markdown("##### An integrated suite of deterministic parsing pipelines, conversational search assistants, and autonomous analytical agents.")
    st.divider()

    # 3-Card Layout using st.columns
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.subheader("📊 Autonomous Inventory Analytics Agent")
            st.markdown(
                "Interact with complex inventory datasets via natural language. Features an autonomous "
                "code-generation engine to execute data queries, real-time contextual dictionary search lookups, "
                "and automated plain-English business finding summaries via an intuitive interactive chat interface."
            )
            # Launch trigger switches programmatically to targeted page
            if st.button("Launch Inventory Agent", type="secondary", use_container_width=True):
                st.switch_page("tasks/task1_inventory_agent.py")

    with col2:
        with st.container(border=True):
            st.subheader("💬 Context-Aware Document Support Assistant")
            st.markdown(
                "Advanced multi-turn support assistant engineered to maintain conversational context across "
                "long sessions. Intelligently tracking user query threads, the assistant proactively handles "
                "fluid topic switching, eliminates redundant answers, and enforces strict, verifiable section citations "
                "for every document resource."
            )
            if st.button("Launch Support Assistant", type="secondary", use_container_width=True):
                st.switch_page("tasks/task2_support_assistant.py")

    with col3:
        with st.container(border=True):
            st.subheader("📄 Intelligent Document Processing (IDP) Pipeline")
            st.markdown(
                "Production-grade hybrid IDP framework designed to classify and extract critical entity structures "
                "from mixed identity proofs and insurance forms. Features advanced text-line spatial heuristic "
                "grouping, deterministic regex validations, and automated multimodal exception routing for handling "
                "challenging handwritten fields."
            )
            if st.button("Launch IDP Pipeline", type="primary", use_container_width=True):
                st.switch_page("tasks/task3_idp_pipeline.py")

# Define pages dictionary mapping task modules
pages = {
    "Suite Hub": [
        st.Page(show_landing_hub, title="Dashboard Home", icon="🏠")
    ],
    "AI & IDP Toolset": [
        st.Page("tasks/task1_inventory_agent.py", title="Inventory Agent", icon="📊"),
        st.Page("tasks/task2_support_assistant.py", title="Support Assistant", icon="💬"),
        st.Page("tasks/task3_idp_pipeline.py", title="IDP Extraction Pipeline", icon="📄")
    ]
}

pg = st.navigation(pages)
pg.run()
