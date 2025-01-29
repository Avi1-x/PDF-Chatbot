import os
import io
import torch
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import tempfile
from sentence_transformers import SentenceTransformer
import pymupdf
import faiss
import numpy as np
import re
from llama_cpp import Llama
import nltk
import gc

# Check if the 'punkt' resource is already downloaded
if not os.path.exists(nltk.data.find('tokenizers/punkt')):
    nltk.download('punkt')
if not os.path.exists(nltk.data.find('tokenizers/punkt_tab')):
    nltk.download('punkt_tab')

# Defining stop words
stop_tokens = ["[INST]", "[/INST]", "User:", "</s>", "<s>", "[Out]", "Note"]

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_file.seek(0)
    pdf_data = io.BytesIO(pdf_file.read())
    with pymupdf.open(stream=pdf_data, filetype="pdf") as pdf:
        text = ""
        for page in pdf:
            text += page.get_text()
    return text

# Function to perform basic pre-processing on text
def process_text(text):
    text = text.lower()
    return text

# Function to split text into chunks (requires special characters to distinguish sentence boundaries)
def split_text_into_chunks(text, chunk_size=200):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())  # Assuming word count as a proxy for token count
        if current_chunk_size + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = sentence_length
        else:
            current_chunk.append(sentence)
            current_chunk_size += sentence_length

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to generate embeddings
def generate_embeddings(chunks, model):
    with torch.no_grad():
        return model.encode(chunks, convert_to_tensor=False)  # Returns numpy array directly
    
# Function to create a FAISS index
def create_faiss_index(embeddings, n_neighbors=3):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # Use brute-force L2 search (exact search)
    index.add(embeddings)
    return index

# Function to query FAISS index
def query_index(index, query, model, chunks, top_k=3):
    with torch.no_grad():
        query_embedding = model.encode([query], convert_to_tensor=False)  # Get NumPy array directly
        distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# Initialize Streamlit app
st.set_page_config(layout='wide')

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "llama" not in st.session_state:
    st.session_state.llama = Llama(
        model_path="path_to_your_local_model/Llama3.2-3B-Instruct_Q8_0.gguf",
        verbose=False, n_batch=1024, n_gpu_layers=6, n_ctx=4096
    )

# Load embedding model
embedding_model = SentenceTransformer("paraphrase-minilm-l6-v2")

# Custom CSS for clean design
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        .st-chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
        
        .stChatInput {
          position: fixed;
          bottom: 3rem;
        }

        .pdf-container {
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        iframe {
            width: 100%;
            height: 400px;
            border: none;
        }
        
        .stFileUploader label {
            text-align: center;
            display: block;
        }
    </style>
""", unsafe_allow_html=True)

# Setting up the title and reset button
button_col, title_col = st.columns([1, 8], vertical_alignment="bottom", gap="small")  # Adjust column ratio as needed

with button_col:
    if st.button("🔄 Reset App"):
        for key in st.session_state.keys():
            del st.session_state[key]
        gc.collect()  # Force garbage collection
        st.rerun()

with title_col:
    st.markdown("<h1 style='text-align: center;'>Interactive PDF Chatbot</h1>", unsafe_allow_html=True)

# Define columns for layout
col1, col2 = st.columns([2, 1])  # More space for chat

with col1:
    # Create box for user input
    user_input = st.chat_input("Ask a question about the document...")

    # Create a container for the chat messages
    chat_container = st.container(height=500)

    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input handling
        if user_input:
            # Check if a PDF is uploaded and processed
            if st.session_state.faiss_index is None or st.session_state.chunks is None:
                # Display an error message if no PDF is uploaded or processed
                with st.chat_message("assistant"):
                    st.error("Please upload a PDF before asking a question.")
            else:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Retrieve context from FAISS
                document_context = ""
                with st.spinner("Retrieving relevant context..."):
                    retrieved_chunks = query_index(
                        st.session_state.faiss_index, user_input, embedding_model, st.session_state.chunks
                    )
                    document_context = " ".join(retrieved_chunks)

                # Full context
                processed_user_input = process_text(user_input)
                full_context = f"Document Context:{document_context}User Query:{processed_user_input}\nPlease provide a concise answer."

                # Debug: Print the full context being sent to the LLM
                print("\nMessage being sent to the LLM:")
                print(full_context)

                # Generate assistant's response
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    response_text = ""

                    for token in st.session_state.llama.create_chat_completion(
                        messages=[{"role": "user", "content": full_context}],
                        stream=True,
                        temperature=0.05,
                        top_k=5,
                        top_p=0.7,
                        max_tokens=256,
                        stop=stop_tokens
                    ):
                        content = token["choices"][0]["delta"].get("content", "")
                        response_text += content
                        response_container.markdown(response_text)

                    # Add assistant's response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})      
                          
with col2:
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        # Display PDF preview
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name

        # st.markdown("<h4 style='text-align: center;'>PDF Preview</h4>", unsafe_allow_html=True)
        pdf_viewer(input=temp_pdf_path, width=1000, height=800, pages_vertical_spacing=10)

        # Process PDF for chunking and embeddings
        pdf_text = extract_text_from_pdf(uploaded_file)
        pdf_text = process_text(pdf_text)
        chunks = split_text_into_chunks(pdf_text)
        embeddings = generate_embeddings(chunks, embedding_model)
        faiss_index = create_faiss_index(embeddings)

        st.session_state["faiss_index"] = faiss_index
        st.session_state["chunks"] = chunks