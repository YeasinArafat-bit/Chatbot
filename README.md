# DocuMind - PDF Chatbot (RAG Demo)

A modern, retrieval-augmented generation (RAG) web application that allows users to upload any PDF document, processes it locally, and answers questions based on the document's content with precise, page-level citations. Powered by Google Gemini models with fallback reliability.

---

## 📸 User Interface

Here is the DocuMind interface in action:

![DocuMind UI Chat Interface](screenshots/screenshot1.png)

![DocuMind UI Document Ingestion](screenshots/screenshot2.png)

---

## ✨ Features

1. **Vibrant & Responsive UI:** Designed with a modern, dark-themed indigo aesthetic using glassmorphism and glowing micro-animations.
2. **Robust Hybrid Retrieval:** Combines Semantic Vector Search (Chroma DB) with BM25 Keyword Search to capture both conceptual meanings and exact matches (e.g. values, names).
3. **Smart Re-upload Handling:** Computes file hashes to instantly load previously processed documents without parsing or re-embedding them.
4. **Model Fallback for High Reliability:** Attempts query answering via a primary model (`gemini-2.5-flash`), falling back automatically to a secondary model if necessary.
5. **Page-Level Citations:** Displays precise source page citations (e.g., *Page 3*, *Page 8*) below assistant replies.
6. **Scanned Document Detection:** Validates whether a PDF contains selectable text and alerts the user if OCR is needed.

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/YeasinArafat-bit/Chatbot.git
cd Chatbot
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open `.env` and enter your Google Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

---

## 🛠️ Built With

* **Frontend:** [Streamlit](https://streamlit.io/) - For building interactive web UIs in Python.
* **LLM API:** [Google Gemini API](https://ai.google.dev/) - Powering intelligent QA and fallbacks (`gemini-2.5-flash`).
* **Framework:** [LangChain](https://www.langchain.com/) - Orchestrating document chunking, prompt engineering, and retriever logic.
* **Vector Store:** [Chroma DB](https://www.trychroma.com/) - A lightweight, local vector database for storing text embeddings.
* **Embeddings:** [sentence-transformers](https://huggingface.co/sentence-transformers) - Running local embeddings (`all-MiniLM-L6-v2`) on-device.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

