# RAG Legal AI Assistant  

A Retrieval-Augmented Generation (RAG) agent for analyzing **legal contracts and documents**, powered by **LangChain, LangGraph, ChromaDB, SentenceTransformers, and Google Gemini**.  

This assistant is designed to give **accurate, citation-backed answers** from uploaded PDFs, while following strict rules to avoid hallucination.  

---

## Features  

- **Document Ingestion**  
  - Load and process **PDF files**.  
  - Automatically split into **chunks with metadata** (page numbers, chunk IDs, sources).  
  - Persist data in a **Chroma vector store** for fast retrieval.  

- **Embeddings**  
  - Uses `multi-qa-MiniLM-L6-cos-v1` SentenceTransformer for dense embeddings.  

- **Vector Database**  
  - Persistent **ChromaDB** backend for efficient similarity search.  

- **Retrieval Tool**  
  - `retriever_tool(query)` fetches top-k chunks relevant to the user query.  
  - Results include inline citations:  
    ```
    [FILENAME | p.<page> | chunk <id>]
    ```

- **LLM Integration**  
  - Powered by **Google Gemini 2.5 Flash** via `langchain_google_genai`.  
  - Strict system prompt enforces:
    - Always query the retriever tool before answering.  
    - Cite every factual claim.  
    - Handle hypotheticals with structured reasoning.  

- **Agent Orchestration**  
  - Built with **LangGraph**.  
  - Defines an agent loop (`LLM → Retriever → LLM`) until the final answer is ready.  
  - Ensures tool usage before response.  

- **Interactive CLI**  
  - Query documents via terminal.  
  - Exit anytime with `exit` or `quit`.  



