# 📄 Chat with PDF — RAG Application

An AI-powered app that lets you upload any PDF and ask questions about it in natural language. Built using RAG (Retrieval-Augmented Generation) architecture.

🔗 **Live Demo:** https://sravya-pdf-chatbot.streamlit.app

---

## 🎯 What it Does

- Upload any PDF file
- Ask questions about the content
- Get accurate AI-powered answers instantly

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web UI |
| pdfplumber | Extract text from PDF |
| FAISS | Vector database for semantic search |
| Sentence Transformers | Convert text to vectors (embeddings) |
| Groq LLM (LLaMA 3) | Generate answers |

---

## 🧠 How it Works (RAG Pipeline)

```
PDF Uploaded
     ↓
Extract Text (pdfplumber)
     ↓
Split into Chunks
     ↓
Convert to Vectors (Sentence Transformers)
     ↓
Store in FAISS Vector Database
     ↓
User asks a Question
     ↓
Search relevant chunks in FAISS
     ↓
Send chunks + question to Groq LLM
     ↓
Answer displayed to user ✅
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/vojjala-sravya/pdf-chatbot.git
cd pdf-chatbot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Get a free key at https://console.groq.com

Create a file `.streamlit/secrets.toml` and add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

**4. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` ✅

---

## 📁 Project Structure

```
pdf-chatbot/
├── app.py               # Main application code
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 💡 Features

- ✅ Upload any PDF
- ✅ Automatic text extraction
- ✅ Semantic search using vector embeddings
- ✅ Fast answers using Groq's free LLM API
- ✅ Shows source chunks used for answer
- ✅ Deployed live on Streamlit Cloud

---

## 🔮 Future Improvements

- [ ] Support multiple PDFs at once
- [ ] Chat history / memory
- [ ] Support for scanned PDFs (OCR)
- [ ] Download answers as text file

---

## 👩‍💻 Author

**Sravya Vojjala**


---

## 📜 License

MIT License — free to use and modify
