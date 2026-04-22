# RAG Customer Support Assistant with HITL

This repository contains the working implementation of a Retrieval-Augmented Generation (RAG) Customer Support Bot. It is orchestrated using LangGraph to implement a stateful workflow, including conditional routing and Human-in-the-Loop (HITL) escalation.

## System Requirements
* Python 3.10+
* A valid Google Gemini API Key

## Setup Instructions

1. **Create and Activate a Virtual Environment:**
   For Windows:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
   For Mac/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies:**
   Install all required libraries for the data pipeline and workflow orchestration:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google API key securely:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```

## Execution Flow

### 1. Data Ingestion (Vector Database Setup)
Before running the bot, you must process the knowledge base to build the local vector database. Ensure your target document is named `sample.pdf` and is located in the root directory.

Run the ingestion script:
```bash
python ingest.py
```
*Action: This script loads the PDF, splits it into semantic chunks, generates vector embeddings using `gemini-embedding-001`, and stores them locally in a `chroma_db` folder.*

### 2. Run the AI Workflow
Once the database is built, you can start the interactive customer support bot:
```bash
python graph_bot.py
```
*Action: This script initializes the LangGraph state machine. It will search the database and automatically answer queries found within the PDF context using `gemini-2.5-flash`. If a query falls outside the knowledge base, the graph conditionally routes the flow to pause execution and prompts the terminal for a human agent's manual response (HITL).*