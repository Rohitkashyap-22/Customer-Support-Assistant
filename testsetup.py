import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
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
    print(f"\n[System] Searching knowledge base for: '{query}'...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # If the bot doesn't know, we force it to output the exact word "ESCALATE"
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
    
    response = chain.invoke({"context": context, "question": query}).content.strip()
    
    # Check if we need to escalate to a human
    if "ESCALATE" in response:
        return {"answer": "", "needs_human": True}
    else:
        return {"answer": response, "needs_human": False}

def human_node(state: BotState):
    """Pauses the system and asks a human for input."""
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

workflow.add_node("rag_node", rag_node)
workflow.add_node("human_node", human_node)

workflow.set_entry_point("rag_node")

workflow.add_conditional_edges("rag_node", route_query)
workflow.add_edge("human_node", END)

app = workflow.compile()

# ==========================================
# 5. INTERACTIVE CHAT (For Live Demo!)
# ==========================================
if __name__ == "__main__":
    print("=== 🚀 Bank of India Credit Card Support Bot Online ===")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input("\n🧑‍💻 You: ")
        
        if user_query.lower() == 'exit':
            print("Shutting down system...")
            break
            
        if not user_query.strip():
            continue
            
        # Run the LangGraph workflow
        result = app.invoke({"query": user_query, "answer": "", "needs_human": False})
        
        # Display the final answer
        print(f"\n🤖 Bot Response: {result['answer']}")