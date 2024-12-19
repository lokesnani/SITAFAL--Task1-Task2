import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from langchain.llms import OpenAI
import streamlit as st

#  Extract text from specific pages using PyPDF2
def extract_text_from_pdf(pdf_path, page_numbers=None):
    reader = PdfReader(pdf_path)
    text = ""
    if page_numbers:
        for page_num in page_numbers:
            text += reader.pages[page_num].extract_text()
    else:
        for page in reader.pages:
            text += page.extract_text()
    return text

# Chunk text for better granularity
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed chunks using SentenceTransformer
def embed_chunks(chunks, model):
    return model.encode(chunks)

# Embed user query
def embed_query(query, model):
    return model.encode([query])

# Retrieve relevant chunks from FAISS
def retrieve_relevant_chunks(query_embedding, index, chunks, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Generate a response using an LLM
def generate_response(chunks, query, llm):
    context = "\n\n".join(chunks)
    prompt = f"Here is the context from the PDF:\n{context}\n\nAnswer the question:\n{query}"
    return llm(prompt)

# RAG pipeline to handle queries
def rag_pipeline(pdf_path, query, model, llm, page_numbers=None, top_k=5):
    text = extract_text_from_pdf(pdf_path, page_numbers)
    if not text.strip():
        return "No text could be extracted from the PDF. Please check the file."

    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, model)

    # Initialize FAISS index
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Query handling
    query_embedding = embed_query(query, model)
    retrieved_chunks = retrieve_relevant_chunks(query_embedding, index, chunks, top_k)

    # Generate and return response
    return generate_response(retrieved_chunks, query, llm)

# Streamlit app
st.title("PDF Chat with RAG Pipeline")
st.subheader("Interact with PDF data and get accurate responses")

# Load embedding model and LLM
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm = OpenAI(temperature=0, api_key="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")  
    return embedding_model, llm

embedding_model, llm = load_models()

# File uploader for PDFs
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Enter your query:")
page_numbers_input = st.text_input("Enter specific page numbers (comma-separated, optional):")

if uploaded_pdf:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    pdf_path = "uploaded_file.pdf"

    if query:
        try:
            # Parse page numbers if provided
            if page_numbers_input:
                page_list = [int(num.strip()) - 1 for num in page_numbers_input.split(",") if num.strip().isdigit()]
            else:
                page_list = None

            st.info("Processing your request, please wait...")
            response = rag_pipeline(pdf_path, query, embedding_model, llm, page_numbers=page_list)
            st.subheader("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query to ask a question about the PDF.")
else:
    st.warning("Please upload a PDF file to proceed.")
