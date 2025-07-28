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
from langchain.prompts import PromptTemplate

# Load environment variables
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

def process_pdf(pdf_files):
    """Extract and split text from one or more PDF files."""
    all_text = ""
    for pdf_file in pdf_files:
        if pdf_file.size == 0:
            st.warning(f"File '{pdf_file.name}' is empty and will be skipped.")
            continue
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.read())
            tmp.flush()
            try:
                reader = PdfReader(tmp.name)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text
            except Exception as e:
                st.error(f"Failed to read file '{pdf_file.name}': {e}")
    if not all_text.strip():
        st.error("No extractable text found in the uploaded files.")
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(all_text)

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
            # Check if the answer is empty or generic or hallucinated
            if (
                not answer.strip()
                or "I don't have information about that" in answer
                or "Sorry" in answer
                or "As an AI language model" in answer
                or "I do not have access" in answer
                or "I don't know" in answer
            ):
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", "Sorry, I can only answer questions based on the uploaded PDF(s). Please ask something from the document(s)."))
            else:
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", answer))

    for speaker, msg in st.session_state.get("chat_history", []):
        if speaker == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

def main():
    st.title("PDF Q&A Chatbot (RAG)")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["chat", "pdf_uploader"])
    
    # Strict prompt for PDF-only answers
    qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. "
        "Answer ONLY using information from the provided context. "
        "If the answer is not in the context, say: "
        "'Sorry, I can only answer questions based on the uploaded PDF(s). Please ask something from the document(s).' "
        "Do not use any outside knowledge. "
        "\n\nContext:\n{context}\n\nQuestion: {question}"
    )
)

    if page == "chat":
        st.title("Welcome to the chatbot")
        st.write("This is the chat page where you can interact with the chatbot.")
        st.info("Ask questions about the provided PDF document. Type 'exit' to end the chat.")

        create_index(INDEX_NAME)
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": qa_prompt
            }
        )
        continual_chat(vector_store, qa_chain)

    elif page == "pdf_uploader":
        st.title("Welcome to the PDF Uploader")
        st.write("Upload your PDF files here for processing.")
        
        pdf_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

        if pdf_files:
            with st.spinner("Processing PDF(s)..."):
                chunks = process_pdf(pdf_files)
                if chunks:
                    st.success(f"Extracted {len(chunks)} text chunks.")
                    create_index(INDEX_NAME)
                    PineconeVectorStore.from_texts(
                        chunks,
                        embeddings,
                        index_name=INDEX_NAME
                    )
                    st.success(f"Stored {len(chunks)} vectors in Pinecone index: {INDEX_NAME}.")

if __name__ == "__main__":
    main()