import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables (API Key)
load_dotenv()

# ==========================================
# 1. DEFINE THE STATE (Data flowing between nodes)
# ==========================================
class BotState(TypedDict):
    query: str
    answer: str
    needs_human: bool

# ==========================================
# 2. DEFINE THE NODES (The Actions)
# ==========================================
def rag_node(state: BotState):
    """Searches the database and tries to answer using Gemini."""
    query = state["query"]
    print(f"\n[System] Searching knowledge base for: '{query}'...")
    
    # Initialize Embedding model and Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Retrieve relevant chunks from the PDF
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Strict System Prompt to prevent hallucination
    prompt_template = """
    You are a customer support bot for Bank of India Credit Cards. 
    Use ONLY the following context to answer. 
    If the context does not contain the answer, reply EXACTLY with the word: ESCALATE. Do not guess.
    
    Context:
    {context}
    
    User Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    # Get response from Gemini
    response = chain.invoke({"context": context, "question": query}).content.strip()
    
    # Check if we need to escalate to a human
    if "ESCALATE" in response:
        return {"answer": "", "needs_human": True}
    else:
        return {"answer": response, "needs_human": False}

def human_node(state: BotState):
    """Pauses the system and asks a human for input (HITL)."""
    print("\n[System Alert] ⚠️ I don't have this information. Escalating to Human Agent...")
    
    # This simulates a dashboard where a human agent types a response
    human_answer = input(f"User asked: '{state['query']}'\n[Agent Dashboard] Type your manual response here: ")
    
    return {"answer": f"[Human Agent] {human_answer}", "needs_human": False}

# ==========================================
# 3. DEFINE THE ROUTING LOGIC
# ==========================================
def route_query(state: BotState):
    """Decides where to go after the rag_node."""
    if state["needs_human"]:
        return "human_node" # Route to human
    return END # Finish the graph

# ==========================================
# 4. COMPILE THE GRAPH
# ==========================================
workflow = StateGraph(BotState)

# Add our nodes
workflow.add_node("rag_node", rag_node)
workflow.add_node("human_node", human_node)

# Set the starting point
workflow.set_entry_point("rag_node")

# Add the routing edges
workflow.add_conditional_edges("rag_node", route_query)
workflow.add_edge("human_node", END)

# Compile into a working application
app = workflow.compile()

# ==========================================
# 5. INTERACTIVE CHAT (For Live Video Demo)
# ==========================================
if __name__ == "__main__":
    print("---- Bank of India Credit Card Support Bot Online ----")
    print("---- INSTRUCTIONS: If you have any query, type your question below.")
    print("                   If you want to close the system, type 'exit'")
    
    while True:
        # 1. Get User Input with a clear prompt
        user_query = input("\n🧑‍💻 Ask your question (or type 'exit' to stop): ")
        
        # 2. Check for Exit command
        if user_query.lower() == 'exit':
            print("Shutting down system... Bye! 👋")
            break
            
        # Ignore empty inputs
        if not user_query.strip():
            continue
            
        # 3. Run the LangGraph workflow
        result = app.invoke({"query": user_query, "answer": "", "needs_human": False})
        
        # 4. Display the final answer (either from Bot or Human)
        print(f"\n🤖 Bot Response: {result['answer']}")