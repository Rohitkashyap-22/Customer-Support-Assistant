import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load the API keys from the .env file
load_dotenv()

# Verify the key is loaded
if not os.environ.get("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY not found. Check your .env file!")
    exit()

def test_llm():
    print("Connecting to Gemini...")
    
    # 2. Initialize the Gemini LLM
    # Updated to gemini-2.5-flash because 1.5 was retired
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 3. Ask a test question
    response = llm.invoke("Say 'System setup is complete and ready for RAG!'")
    
    # 4. Print the response
    print("\nResponse from AI:")
    print("-" * 30)
    print(response.content)
    print("-" * 30)

if __name__ == "__main__":
    test_llm()