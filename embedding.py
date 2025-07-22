
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


JSON_INPUT_PATH = "data/data_chunks.json"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load chunks from JSON
def load_json_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    chunks = []
    for item in json_data:
        text = item.pop("text", "")
        metadata = item
        chunks.append(Document(page_content=text, metadata=metadata))
    return chunks

# Initialize embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Save documents into FAISS vectorstore
def save_to_faiss(docs, model, save_path):
    db = FAISS.from_documents(docs, model)
    db.save_local(save_path)
    print(f"Saved {len(docs)} chunks to vectorstore at {save_path}")

# Main function
def main():
    if Path(JSON_INPUT_PATH).exists():
        print(f"Loading from JSON: {JSON_INPUT_PATH}")
        docs = load_json_chunks(JSON_INPUT_PATH)
        print(f"Loaded {len(docs)} chunks")
    else:
        print("JSON file not found!")
        return

    model = get_embedding_model()
    save_to_faiss(docs, model, DB_FAISS_PATH)

if __name__ == "__main__":
    main()
