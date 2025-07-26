from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # Return the text content of all pages concatenated
    return "\n".join(doc.page_content for doc in documents)