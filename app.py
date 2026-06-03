import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ─────────────────────────────────────────────
# CONFIG — paste your Groq API key here
# Get it free at: https://console.groq.com
# ─────────────────────────────────────────────
GROQ_API_KEY = GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ─────────────────────────────────────────────
# STEP 1: Extract text from PDF
# ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# ─────────────────────────────────────────────
# STEP 2: Split text into small chunks
# Think of this like cutting a book into pages
# ─────────────────────────────────────────────
def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap   # overlap so we don't miss context
    return chunks


# ─────────────────────────────────────────────
# STEP 3: Convert chunks to vectors & store in FAISS
# Vectors = numbers that capture meaning of text
# FAISS = fast search database for these vectors
# ─────────────────────────────────────────────
def build_vector_store(chunks, model):
    embeddings = model.encode(chunks)                        # convert text → numbers
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]                          # size of each vector
    index = faiss.IndexFlatL2(dimension)                     # create FAISS index
    index.add(embeddings)                                    # store all vectors

    return index, embeddings


# ─────────────────────────────────────────────
# STEP 4: Find most relevant chunks for a question
# ─────────────────────────────────────────────
def search_relevant_chunks(question, index, chunks, model, top_k=3):
    question_vector = model.encode([question]).astype("float32")
    distances, indices = index.search(question_vector, top_k)   # find closest chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


# ─────────────────────────────────────────────
# STEP 5: Ask Groq LLM with relevant context
# ─────────────────────────────────────────────
def ask_llm(question, context_chunks):
    client = Groq(api_key=GROQ_API_KEY)

    context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I couldn't find this in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions about it!")

# Load the embedding model once (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:

    with st.spinner("Reading and processing your PDF..."):
        # Run all 3 steps
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = split_into_chunks(raw_text)
        index, embeddings = build_vector_store(chunks, model)

    st.success(f"✅ PDF processed! Found {len(chunks)} chunks.")

    # Chat section
    st.divider()
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            relevant_chunks = search_relevant_chunks(question, index, chunks, model)
            answer = ask_llm(question, relevant_chunks)

        st.markdown("### 💬 Answer")
        st.write(answer)

        # Show source chunks (optional, good for debugging)
        with st.expander("📚 Source chunks used"):
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(chunk)
                st.divider()
