import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import matplotlib.pyplot as plt
from chromadb.config import Settings

# Initialize embedding model and LLM
embedding_function = HuggingFaceEmbeddings(model_name="distilbert-base-nli-stsb-mean-tokens")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Function to process PDFs
def process_pdfs(pdf_files):
    """
    Processes uploaded PDF files, extracts text, and generates embeddings.
    """
    all_documents = []
    metadata = []

    # Persistent ChromaDB setup
    chroma_client = Chroma(
        collection_name="form_10k_docs",
        embedding_function=embedding_function,
        persist_directory="./chroma_db"
    )

    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            all_documents.append(chunk.page_content)
            metadata.append({"document_name": os.path.basename(pdf_path), "page_number": chunk.metadata["page"]})

    chroma_client.add_texts(all_documents, metadatas=metadata)
    chroma_client.persist()  # Save embeddings to disk
    print(f"Processed {len(all_documents)} chunks.")
    return chroma_client

# Function to generate comparison
def generate_comparison(pdf_files, prompt):
    """
    Uses the local LLM to generate a comparison between documents.
    """
    if not pdf_files or len(pdf_files) < 2:
        return "Please upload at least two PDF documents for comparison.", ""

    # Process PDFs
    chroma_client = process_pdfs(pdf_files)

    # Retrieve relevant chunks
    docs = chroma_client.similarity_search(prompt, k=3)
    print(f"Retrieved {len(docs)} documents from Chroma.")
    if not docs:
        return "No relevant documents found. Please try a different prompt.", ""

    retrieved_texts = "\n".join([doc.page_content for doc in docs])

    # Generate insights with LLM
    input_text = f"Prompt: {prompt}\nRelevant Texts:\n{retrieved_texts}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(**inputs, max_length=300)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response, retrieved_texts

# Streamlit App
st.title("📄 Content Engine: Analyze and Compare PDF Documents.")
st.write(
    "Upload two or more PDF files and provide a prompt to generate comparisons."
)

# File Upload
uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
prompt = st.text_input("Enter your comparison prompt",
                       placeholder="e.g., 'Compare the financial performance of Alphabet and Tesla.'")

if st.button("Generate Comparison"):
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            file_paths.append(file_path)

        # Generate the comparison
        result, retrieved_texts = generate_comparison(file_paths, prompt)

        # Display the output
        st.text_area("Comparison Insights", value=result, height=300)

        # Generate and display line graph for retrieved texts
        if retrieved_texts:
            doc_labels = [f"Doc {i+1}" for i in range(len(retrieved_texts.split('\n')))]
            text_lengths = [len(text) for text in retrieved_texts.split('\n')]

            fig, ax = plt.subplots(figsize=(20, 10))  # Adjusted the width and height
            ax.plot(doc_labels, text_lengths, marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=8)
            ax.set_title("Text Length of Retrieved Documents", fontsize=16, weight='bold')
            ax.set_xlabel("Documents", fontsize=10)
            ax.set_ylabel("Text Length (characters)", fontsize=10)
            st.pyplot(fig)

        # Clean up temporary files
        for file_path in file_paths:
            os.remove(file_path)
