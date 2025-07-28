import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
INDEX_NAME = "pdf-chatbot-index"

def create_index(INDEX_NAME):
    existing_indexes = [index.name for index in pc.list_indexes()] if pc.list_indexes() else []
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        st.info(f"Creating new index: {INDEX_NAME}...")
    return pc.Index(INDEX_NAME)

def process_pdf(pdf_file):
    """Extract and split PDF text"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(pdf_file.read())
        reader = PdfReader(tmp.name)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def continual_chat(vector_store, qa_chain):
    st.info("Type 'exit' to end the chat.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("You:", key=f"chat_input_{len(st.session_state.chat_history)}")
    if user_input:
        if user_input.lower() == "exit":
            st.info("Chat ended. Upload a new PDF to start again.")
            st.session_state.chat_history = []
        else:
            result = qa_chain({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]
            if answer.strip():
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", answer))
            else:
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", "Sorry, I can only answer questions based on the uploaded PDF. Please ask something from the document."))
    for speaker, msg in st.session_state.get("chat_history", []):
        if speaker == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

def main():
    st.title("PDF Q&A Chatbot (RAG)")
    st.info("Ask questions about the provided PDF document. Type 'exit' to end the chat.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    if pdf_file:
        with st.spinner("Processing PDF..."):
            chunks = process_pdf(pdf_file)
            st.success(f"Extracted {len(chunks)} text chunks")
            create_index(INDEX_NAME)
            PineconeVectorStore.from_texts(
                chunks,
                embeddings,
                index_name=INDEX_NAME
            )
            st.success(f"Stored {len(chunks)} vectors in Pinecone index: {INDEX_NAME} ")
        vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        continual_chat(vector_store, qa_chain)

if __name__ == "__main__":
    main()