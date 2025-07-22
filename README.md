# RAG_Based_AI_Assistant_For_Specific_Json_File
This project is a Retrieval-Augmented Generation (RAG) based AI Assistant built using LangChain, FAISS, and HuggingFace Transformers. It is specifically designed to answer user queries based on a given structured JSON file that contains document chunks. The assistant retrieves the most relevant chunks from a vector database and generates context-aware responses using Groq-hosted LLMs. A Streamlit interface is provided for a simple chat-based interaction.

## Features
Parses and indexes structured JSON document chunks

Stores embeddings using FAISS vector database

Loads vectorstore and uses similarity search for document retrieval

Uses HuggingFace sentence-transformer embeddings

Supports LLM-based question answering with Groq API

Custom prompt template for better control over generated answers

Provides source citations including PDF name, section heading, and link

Easy-to-use Streamlit chat interface

## Folder Structure

.
├── app.py                    # Streamlit chat UI
├── embedding_pipeline.py     # Script to embed and store documents in FAISS
├── data/
│   └── data_chunks.json      # Input JSON with document chunks and metadata
├── vectorstore/
│   └── db_faiss/             # Saved FAISS vectorstore directory
├── .env                      # Contains GROQ_API_KEY
└── requirements.txt          # Python dependencies
## How It Works
Embedding Pipeline
The embedding_pipeline.py script reads the data/data_chunks.json file

Each JSON item is converted into a LangChain Document object with metadata

Sentence-transformer model from HuggingFace generates embeddings

FAISS is used to index and save the document vectors locally

## Query Handling
The app.py file launches a Streamlit-based chatbot interface

User inputs a question via the chat input

Top relevant chunks are retrieved from the FAISS vectorstore

A custom prompt is combined with the retrieved context

The query is passed to a Groq-hosted LLM (e.g., LLaMA 4)

The assistant returns an accurate answer along with document citations

## Installation
Clone the repository:


git clone https://github.com/your-username/RAG_Based_AI_Assistant_For_Specific_Json_File.git
cd RAG_Based_AI_Assistant_For_Specific_Json_File
## (Optional) Create a virtual environment:


python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
## Install dependencies:


pip install -r requirements.txt
## Environment Variables
Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY=your_groq_api_key_here
## Usage
### Step 1: Embed the JSON file
python embedding.py
This command will generate and store embeddings in the vectorstore/db_faiss directory.

### Step 2: Run the Streamlit chatbot
streamlit run app.py
This will open a local Streamlit interface for asking questions based on the JSON content.

## Requirements
Refer to requirements.txt. Key packages include:

langchain

langchain-community

langchain-core

langchain-groq

faiss-cpu

sentence-transformers

streamlit

python-dotenv

## Technologies Used
LangChain for document parsing and QA chain

FAISS for vector indexing and similarity search

HuggingFace Transformers for sentence embeddings

Groq API for high-performance LLM inference

Streamlit for user interface
