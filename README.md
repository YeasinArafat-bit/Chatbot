# DocuMind - PDF Chatbot (RAG Demo)

A modern, retrieval-augmented generation (RAG) web application that allows users to upload any PDF document, processes it locally, and answers questions based on the document's content with precise, page-level citations. Powered by Google Gemini models with fallback reliability.

---

## Features

1. **Vibrant & Responsive UI:** Designed with a modern, dark-themed indigo aesthetic (using glassmorphism, glowing pulse micro-animations, and clean typography).
2. **Robust RAG Pipeline:** Extracts document content, splits it into chunks preserving page metadata, generates high-quality embeddings, and stores them in a local Chroma vector database.
3. **Smart Re-upload Handling:** Computes file hashes to instantly load previously processed documents without parsing or re-embedding them.
4. **Conversational Memory:** Uses chat history to rephrase follow-up questions to query the vector store contextually.
5. **Model Fallback for High Reliability:** Attempts query answering via a primary model (`gemini-2.5-flash`), falling back automatically to a secondary model (`gemini-2.0-flash`) if necessary, logging all execution details.
6. **Graceful Scanned Document Validation:** Detects whether a PDF is an image-only/scanned document (lacks selectable text) and alerts the user with helpful tips.
7. **Page-Level Citations:** Displays precise source page citations (e.g., *Page 3*, *Page 8*) below assistant replies.

---

## Project Structure

```text
pdf-chatbot/
├── app.py                  # Streamlit main application UI
├── config.py               # Settings loader (loads from .env)
├── document_processor.py   # PDF loading, scanned validation, chunking, and Chroma DB integration
├── rag_chain.py            # Retrieval and LLM QA chain (with primary/fallback logic)
├── logger.py               # Custom standard logging configuration (logs to app.log and stdout)
├── vector_store/           # Persisted Chroma database folder (Git ignored)
├── .env                    # Local API credentials and configurations (Git ignored)
├── .env.example            # Environment variables placeholder
├── .gitignore              # Ignored files (virtual envs, local db, credentials, logs)
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.11 or later installed on your system.

### 2. Clone the Repository
Clone the project repository to your local machine and navigate into the project directory:
```bash
git clone <repository-url>
cd Chatbot
```

### 3. Create a Virtual Environment
Create and activate a virtual environment to manage dependencies cleanly:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

### 5. Setup Environment Variables
Copy `.env.example` to a new file named `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your Google Gemini API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## Running the Application Locally

Run the Streamlit application using the command line:
```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

---

## How to Deploy to Streamlit Community Cloud

Deploying your PDF Chatbot to Streamlit Community Cloud is simple:

1. **Push your code to GitHub:**
   Make sure you push your codebase to a public or private GitHub repository. (Verify that `.env`, `vector_store/`, and `venv/` are excluded by `.gitignore`).
2. **Log into Streamlit Community Cloud:**
   Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Deploy the App:**
   - Click **"New app"**.
   - Select your repository, branch, and specify `app.py` as the main file path.
   - Click **"Deploy"**.
4. **Configure Secrets:**
   - In your app's dashboard, go to **Settings** -> **Secrets**.
   - Paste your `.env` contents there (specifically the `GEMINI_API_KEY`):
     ```toml
     GEMINI_API_KEY = "your_actual_gemini_api_key"
     ```
   - Streamlit will automatically load these secrets into environment variables.

---

## 🔒 Privacy & Vector Storage Note

> [!IMPORTANT]
> **Data Storage Privacy Notice:**
> When you upload a PDF, this application processes its content and creates vector representations (embeddings) of the text. These embeddings, along with the text segments, are stored locally on your machine in the folder specified by `VECTOR_STORE_PATH` (default: `./vector_store/`).
> 
> * No documents or embeddings are uploaded to external database providers.
> * Your document contents are sent to Google Gemini APIs only to perform embeddings and to generate answers.
> * **Recommendation:** For maximum security when handling sensitive or confidential documents, we recommend periodically deleting the contents of the `vector_store/` directory on your local device.

---

## Screenshots

Below is a conceptual layout of the application dashboard:

![DocuMind UI Dashboard Mockup](https://via.placeholder.com/1000x600/0f172a/ffffff?text=DocuMind+UI+Dashboard+Mockup)
