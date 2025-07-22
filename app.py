# app.py

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def main():
    st.set_page_config(page_title="Research Chatbot", layout="centered")
    st.title("AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Ask your question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        CUSTOM_PROMPT = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know, say "I don't know" — don't guess.

        Context: {context}
        Question: {question}

        Answer:
        """

        try:
            vectorstore = get_vectorstore()
            if not vectorstore:
                st.error("Vectorstore not loaded.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.2,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT)}
            )

            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            docs = result["source_documents"]

            citation_text = "\n\n### Citations:\n"
            for doc in docs:
                meta = doc.metadata
                source_id = meta.get("source_doc_id", "unknown.pdf")
                heading = meta.get("section_heading", "Unknown Section")
                link = meta.get("link", meta.get("source", "#"))
                citation_text += f"- **{source_id}** | *{heading}* | [Link]({link})\n"

            final = answer + citation_text
            st.chat_message("assistant").markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
