# shared/llm.py

## Role in the Project
In a Modular AI Platform, you want to avoid initializing your LLM clients inside every single feature script. If you decide to change models, update an API endpoint, or switch from a cloud provider to a local provider, doing it in multiple places introduces major bugs and code duplication.

shared/llm.py acts as the central nervous system for model access. By providing a single, standardized factory function (get_llm()), all three applications—the Excel Data Agent, the RAG Assistant, and the Document Extraction Pipeline—import this unified interface. This design abstracts away backend complexity and ensures uniform generation settings (like creativity controls) across the entire codebase.

The Switch/Fallback Comment: The initial block outlines a clean migration path. If you transition from cloud inference to a completely local setup, you only need to swap two lines here using ChatOllama(model="mistral"), instantly routing the entire multi-app suite to your local hardware.


# shared/embeddings.py
### Role in the Project
This file is the mathematical core for Task 2 (Document-aware Support Assistant). Before your RAG (Retrieval-Augmented Generation) chatbot can search through PDFs, the text needs to be translated into a format a computer can understand (numbers). This utility handles that translation. By keeping it in the shared/ folder, you also leave the door open for Task 3 to use embeddings for document classification later on.

### Core Architectural & AI Theory
The Singleton Pattern: This is the most important concept in this file. Loading a machine learning model into memory takes time and consumes RAM. If your Streamlit app re-loaded this model every time a user asked a question, the app would crash or freeze. The Singleton pattern ensures the model is loaded exactly once and reused globally.

Why all-MiniLM-L6-v2?: You specifically chose this model because it is the industry standard for local, fast, and free embeddings. It generates 384-dimensional vectors, which are small enough to process rapidly on a CPU but dense enough to capture deep semantic meaning.


# task 3 
In Intelligent Document Processing (IDP), this is called Straight-Through Processing (STP) with AI Exception Handling. By using deterministic rules (Regex/Spatial) for 80% of the documents and only routing the failed 20% to Gemini, you get the best of both worlds: lightning-fast processing, near-zero hallucination risk for standard fields, and drastically lower API compute costs, while still saving human operators from manual data entry.