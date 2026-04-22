import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma # Updated to fix the warning!
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

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
    """Searches the database and tries to answer."""
    query = state["query"]
    print(f"\n[Bot] Let me check the documentation for: '{query}'...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # If the bot doesn't know, we force it to output the exact word "ESCALATE"
    prompt_template = """
    You are a customer support bot. Use ONLY the following context to answer. 
    If the context does not contain the answer, reply EXACTLY with the word: ESCALATE. Do not guess.
    
    Context:
    {context}
    
    User Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "question": query}).content.strip()
    
    # Check if we need to escalate to a human
    if "ESCALATE" in response:
        return {"answer": "", "needs_human": True}
    else:
        return {"answer": response, "needs_human": False}

def human_node(state: BotState):
    """Pauses the system and asks a human for input."""
    print("\n[System Alert] ⚠️ I don't know the answer. Escalating to Human Agent...")
    # This simulates a dashboard where a human agent types a response
    human_answer = input(f"User asked: '{state['query']}'\nType your manual response here: ")
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
# 5. TEST THE WORKFLOW
# ==========================================
if __name__ == "__main__":
    print("=== 🚀 Customer Support Bot Online ===")
    
    # TEST 1: An in-domain question (Bot should answer automatically)
    print("\n--- Test 1: Known Query ---")
    result1 = app.invoke({"query": "What is the main objective of this project?", "answer": "", "needs_human": False})
    print(f"\nFinal Response Sent to User:\n{result1['answer']}")
    
    print("\n" + "="*50)
    
    # TEST 2: An out-of-domain question (Should trigger HITL escalation)
    print("\n--- Test 2: Unknown Query ---")
    result2 = app.invoke({"query": "What is your company's refund policy?", "answer": "", "needs_human": False})
    print(f"\nFinal Response Sent to User:\n{result2['answer']}")