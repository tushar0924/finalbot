import streamlit as st
from transformers import pipeline
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from pypdf import PdfReader
import re
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from io import BytesIO



# Set the Hugging Face API key
os.environ["HF_API_KEY"] = "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"

# Hugging Face pipeline setup
@st.cache_resource(show_spinner=False)
def get_text_generation_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    text_generation_pipeline = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        truncation=True,
        pad_token_id=50256,
        device=device
    )
    print(f"Device selected: {device}")
    return text_generation_pipeline

# Cached function to create a vectordb for the provided PDF files
@st.cache_data
def create_vectordb(files, filenames):
    with st.spinner("Creating vector database..."):
        print(f"Filenames: {filenames}")
        vectordb = get_index_for_pdf([file.getvalue() for file in files], filenames)
    return vectordb

# Parse PDF
def parse_pdf(file: BytesIO, filename: str):
    print(f"Parsing PDF: {filename}")
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            text = re.sub(r"\n\s*\n", "\n\n", text)
            output.append(text)
    print(f"Extracted text from {filename}: {output[:100]}...")  # Print part of the extracted text for debugging
    return output, filename

# Convert text to documents
def text_to_docs(text: list, filename: str):
    if isinstance(text, str):
        text = [text]
    print(f"Converting text to docs for {filename}")
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust chunk size as needed
            chunk_overlap=100,  # Adjust overlap for context
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename
            doc_chunks.append(doc)
    print(f"Created {len(doc_chunks)} document chunks for {filename}")
    return doc_chunks

# Create FAISS vector index
def docs_to_index(docs):
    print(f"Creating FAISS index with {len(docs)} documents")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index = FAISS.from_documents(docs, embeddings)
    return index

# Get the index for PDF
def get_index_for_pdf(pdf_files, pdf_names):
    print(f"Generating vector index for {len(pdf_files)} PDFs")
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents.extend(text_to_docs(text, filename))
    index = docs_to_index(documents)
    print(f"FAISS index created successfully")
    return index


# Define the template for the chatbot prompt
prompt_template = """
You are a helpful Assistant who answers users' questions based on the context provided.

Use the context to answer the questions.

The PDF content is:
{pdf_extract}

User question: {user_query}
"""

# Streamlit app
st.set_page_config(page_title="PDF Chatbot with Hugging Face", page_icon="🤖")
st.title("PDF Chatbot using Hugging Face")

# Initialize chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Upload PDF files
pdf_files = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

if pdf_files:
    pdf_file_names = [file.name for file in pdf_files]
    print(f"PDF files uploaded: {pdf_file_names}")
    vectordb = create_vectordb(pdf_files, pdf_file_names)
    st.session_state["vectordb"] = vectordb

# Get the user's question
user_query = st.chat_input("Ask a question about the PDF(s)")

if user_query:
    print(f"User query: {user_query}")
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("Please upload PDF(s) first.")
            st.stop()

    # Retrieve related chunks using FAISS similarity search
    search_results = vectordb.similarity_search(user_query, k=3)
    print(f"Search results: {[result.page_content[:100] for result in search_results]}...")  # Print part of the search result
    pdf_extract = "\n".join([result.page_content for result in search_results])

    # Update the prompt with the pdf extract and user query
    prompt = prompt_template.format(pdf_extract=pdf_extract, user_query=user_query)
    print(f"Generated prompt: {prompt[:500]}...")  # Print part of the prompt

    # Use the Hugging Face pipeline to generate a response based on PDF extracts
    text_generation_pipeline = get_text_generation_pipeline()
    response = text_generation_pipeline(prompt, max_new_tokens=500, num_return_sequences=1, temperature=0.7,do_sample=True)[0]['generated_text']
    print(f"Generated response: {response}")

    # Extract the generated response from the output
    # Here, we assume the response does not include the prompt. Adjust if necessary based on the model output.
    response = response.replace(prompt, "").strip()  # Remove prompt part if included

    # Display the conversation
    st.session_state["chat_history"].append({"role": "user", "content": user_query})
    st.session_state["chat_history"].append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.get("chat_history", []):
    role, content = message["role"], message["content"]
    with st.chat_message(role):
        st.write(content)

