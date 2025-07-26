from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.pdf_loader import extract_text_from_pdf

def split_text(pdf_path, chunk_size=1000, chunk_overlap=200):
    extracted_text = extract_text_from_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(extracted_text)