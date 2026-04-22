import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# Updated import right here!
from langchain_core.prompts import PromptTemplate

# 1. Load API Keys
load_dotenv()

def test_rag_query(user_query):
    print(f"Searching database for: '{user_query}'...\n")
    
    # 2. Load the embeddings and connect to our existing ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 3. Retrieve the top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(user_query)
    
    # Combine the chunks into a single block of text (our context)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # 4. Set up the LLM 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 5. Create a strict prompt template to prevent hallucination
    prompt_template = """
    You are a helpful customer support assistant. 
    Use ONLY the following context to answer the user's question. 
    If the context does not contain the answer, say "I don't have enough information to answer that." Do not guess.
    
    Context:
    {context}
    
    User Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Chain them together
    chain = prompt | llm
    
    # 6. Generate the answer
    print("Generating response...\n")
    response = chain.invoke({"context": context, "question": user_query})
    
    print("--- AI RESPONSE ---")
    print(response.content)
    print("-------------------")

if __name__ == "__main__":
    # Feel free to change this question to something specific from your PDF!
    question = "What is the main objective of this project?"
    test_rag_query(question)