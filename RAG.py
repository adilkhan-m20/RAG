import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from IPython.display import Image, display
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
#from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

"""llm = ChatOllama(
    model = "qwen3:8b",
    temperature = 0,
    base_url="http://172.31.48.1:11434"
)"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1,
    api_key=os.getenv("GEMINI_API_KEY")
)

# a wrapper class to make SentenceTransformer compatible with LangChain
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Convert texts to embeddings using SentenceTransformer's encode method
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()  # Convert numpy array to list of lists
    
    def embed_query(self, text: str) -> List[float]:
        # For single query embedding
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()  # Convert numpy array to list
    
embeddings = SentenceTransformerEmbeddings('multi-qa-MiniLM-L6-cos-v1')

pdf_path = input("Enter PDF Path: ")

if pdf_path != "None":
    # check if pdf path exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pdf_loader = PyPDFLoader(pdf_path)

    # check if PDF exists
    try:
        pages = pdf_loader.load()
        print(f"PDF has been loaded and has {len(pages)} pages")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        raise

    # Chunking Process
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 128,
        chunk_overlap = 12,
        add_start_index = True,
    )

    pages_split = text_splitter.split_documents(pages)

    for idx,doc in enumerate(pages_split):
        src = os.path.basename(pdf_path)
        doc.metadata = dict(doc.metadata) if doc.metadata else {}
        doc.metadata["chunk_id"] = idx
        doc.metadata["source"] = src

persist_directory = r"VECTOR_DATABASE"
collection_name = "Vector_DB"

# Check if the vector store already exists
if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
    # --- LOAD MODE: Vector store exists ---
    print("Loading existing vector store from:", persist_directory)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    print("Vector store loaded successfully.")
    print(f"Current number of documents in collection: {vectorstore._collection.count()}")
    if pdf_path != "None":
        try:
            # This part adds new chunks to the already-loaded collection
            vectorstore.add_documents(documents=pages_split)
            print(f"Successfully added {len(pages_split)} new chunks.")
            print(f"New total number of documents: {vectorstore._collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to existing collection: {str(e)}")
            raise
else:
    # --- CREATE MODE: Vector store does not exist ---
    print("No existing vector store found. Creating a new one...")
    # Ensure the directory exists before Chroma tries to create it
    os.makedirs(persist_directory, exist_ok=True) 
    try:
        vectorstore = Chroma.from_documents(
            documents = pages_split,
            embedding = embeddings,
            persist_directory = persist_directory,
            collection_name = collection_name
        )
        print(f"Created Chroma DB vector store")
    except Exception as e:
        print(f"Error setting up ChromaDB: {str(e)}")
        raise

retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":5} # k is amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns information from the vector database"""

    docs = retriever.invoke(query)

    if(not docs):
        return "I found no relevant information in the Vector DB for your query"
    
    results = []
    
    for doc in docs:
        src = doc.metadata.get("source","unknown")
        page = doc.metadata.get("page","n/a")
        chunk = doc.metadata.get("chunk_id","n/a")
        text = doc.page_content
        results.append(f"[source: {src} | page: {page} | chunk: {chunk}] {text}")

    header = f"found {len(results)} for query: \"{query}\"\n\n"
    
    return header + "\n\n".join(results)

tools = [retriever_tool]

llm = llm.bind_tools(tools)

# 1. TypedDict
#A special dictionary with fixed, typed keys (from typing module)
#Provides type checking for dictionary keys and values
#Like a regular dict but with defined structure
# 2. message: Sequence[BaseMessage]
#The dictionary has a key called "message"
#The value must be a Sequence (list, tuple) of BaseMessage objects
# 3. Annotated[..., add_messages]
#Adds metadata/annotation to the type
#The add_messages annotation tells the framework to automatically append new messages to this sequence
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

initial_state: AgentState = {"messages": []}

def should_continue(state: AgentState):
    """Check if the last message contains tool calls"""
    result = state["messages"][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls)>0

system_prompt = """You are an intelligent AI assistant specialized in legal documents loaded into your knowledge base. 
When answering, follow these rules:

1. Use only the vector DB (retriever tool) for claims about the document. For each factual claim taken from the document you MUST include an inline citation in square brackets directly after the claim, formatted like: [FILENAME | p.<page> | chunk <id>]. If you quote verbatim, put the quote in double quotes and include the citation.
2. For user hypotheticals (e.g. 'what if I break clause X?') follow this structured answer:
   a) Locate and quote the relevant clause (with citation).
   b) Explain the likely legal consequences of breaching that clause (identify remedy, liability, penalties, timelines if present in the doc), citing document parts for each claim.
   c) Suggest immediate next steps and possible mitigations (contractual remedies, notice requirements, dispute resolution clauses), citing supporting text.
   d) If the document lacks necessary clarity, explicitly state what is missing and propose suggested clause text.
3. If you can't find anything in the document, say so and do not invent facts.
4. Keep answers concise and cite every place you use the contract text.

You may call the retriever tool multiple times if needed to find clause text or cross-reference other parts of the document.You must always call the retriever_tool before answering any question related to the document.
Never answer directly without using the tool.

"""

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """function to call the LLM with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content = system_prompt)] + messages

    #if len(messages) > 5:
    #messages = [SystemMessage(content = system_prompt)] + messages[-1]

    message = llm.invoke(messages)
    return {"messages" : [message]}

tools_dict = {tool_.name: tool_ for tool_ in tools}

#Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        name = t.get("name")
        args = t.get("args")
        query = args.get("query")
        print(f"Calling tool: {name} with query: {query}")

        if name not in tools_dict:
            print(f"\nTool: {name} does not exist.")
            result = "Incorrect tool name, retry and select tool from list of available tools"
        else:
            # invoke and capture result
            result = tools_dict[name].invoke(query)
            print(f"Result Length: {len(str(result))}")

        tool_call_id = t.get("id") 
        results.append(ToolMessage(tool_call_id=tool_call_id, name=name, content=str(result)))
    
    print("Tools Execution Complete. Back to the model")
    return {"messages": results}

graph = StateGraph(AgentState)
graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent","llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

display(Image(rag_agent.get_graph().draw_mermaid()))

def running_agent():
    print("/n=====RAG AGENT=====")

    while True:
        user_input = input("\nEnter your query: ")
        if(user_input.lower() in ['exit','quit']):
            break

        messages = [HumanMessage(content = user_input)]
        result = rag_agent.invoke({"messages": messages})

        print("\n=====Answer=====")
        print(result["messages"][-1].content)

running_agent()