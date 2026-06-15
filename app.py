import patch_protobuf
import os
import streamlit as st

import config
from logger import get_logger
from document_processor import (
    calculate_file_hash,
    process_pdf,
    get_vector_store
)
from rag_chain import query_rag_chain, query_rag_chain_stream

# Initialize logger
logger = get_logger(__name__)

# Set up page configurations
st.set_page_config(
    page_title="DocuMind - PDF Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
}

/* Gradient Header */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    padding: 2rem;
    border-radius: 16px;
    border-left: 6px solid #6366f1;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.25);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.app-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
    pointer-events: none;
}

.app-title {
    color: #ffffff;
    font-weight: 800;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.app-title span {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.app-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.5rem;
    margin-bottom: 0;
    font-weight: 400;
}

/* Sidebar Title and Pulse Icon */
.sidebar-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
}

.sidebar-title span.title-gradient {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.pulse-circle {
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: #6366f1;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7);
    animation: pulse 1.6s infinite;
    margin-right: 12px;
}

@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 10px rgba(99, 102, 241, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(99, 102, 241, 0);
    }
}

/* Metadata Card */
.meta-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.meta-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}

.meta-item:last-child {
    margin-bottom: 0;
}

.meta-label {
    color: #94a3b8;
}

.meta-value {
    color: #f1f5f9;
    font-weight: 500;
}

/* Citation and badges styling */
.citation-container {
    margin-top: 0.6rem;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.citation-chip {
    background-color: rgba(99, 102, 241, 0.1);
    color: #818cf8;
    border: 1px solid rgba(99, 102, 241, 0.2);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.model-badge {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.chat-banner {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #e2e8f0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Helper function to clear session state
def reset_session():
    st.session_state["messages"] = []
    st.session_state["collection_name"] = None
    st.session_state["active_file_name"] = None
    st.session_state["active_file_size"] = None
    st.session_state["pages_processed"] = None
    st.session_state["db"] = None
    logger.info("Session state reset and cleared.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "collection_name" not in st.session_state:
    st.session_state["collection_name"] = None
if "active_file_name" not in st.session_state:
    st.session_state["active_file_name"] = None
if "active_file_size" not in st.session_state:
    st.session_state["active_file_size"] = None
if "pages_processed" not in st.session_state:
    st.session_state["pages_processed"] = None
if "db" not in st.session_state:
    st.session_state["db"] = None

# Sidebar Design
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title"><span class="pulse-circle"></span>DocuMind <span class="title-gradient">PDF</span></div>',
        unsafe_allow_html=True
    )
    
    # 1. Document Upload Phase (Show in sidebar if no active collection is loaded)
    if not st.session_state["collection_name"]:
        st.markdown("### 📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file to begin",
            type=["pdf"],
            help=f"Maximum file size allowed is {config.MAX_FILE_SIZE_MB}MB."
        )
        
        if uploaded_file:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            # Size validation
            if file_size_mb > config.MAX_FILE_SIZE_MB:
                st.error(
                    f"❌ File size ({file_size_mb:.2f}MB) exceeds the maximum limit of "
                    f"{config.MAX_FILE_SIZE_MB}MB. Please upload a smaller file."
                )
                logger.warning(f"File upload rejected due to size: {uploaded_file.name} ({file_size_mb:.2f}MB)")
            else:
                file_bytes = uploaded_file.read()
                file_hash = calculate_file_hash(file_bytes)
                
                # Formatted file size string
                if file_size_mb < 0.1:
                    size_str = f"{uploaded_file.size / 1024:.2f} KB"
                else:
                    size_str = f"{file_size_mb:.2f} MB"
                    
                logger.info(f"File uploaded successfully: {uploaded_file.name} ({size_str}), Hash: {file_hash}")
                
                with st.spinner("Processing document and generating embeddings..."):
                    try:
                        # Check if Chroma already has this collection stored
                        db_exists = False
                        existing_db = get_vector_store(file_hash, chunks=None)
                        
                        try:
                            existing_count = existing_db._collection.count()
                        except Exception:
                            existing_count = 0
                            
                        if existing_count > 0:
                            db_exists = True
                            db = existing_db
                            
                            # Extract pages processed from existing database metadata
                            existing_docs = existing_db.get()
                            pages = set()
                            for meta in existing_docs.get("metadatas", []):
                                if meta and "page_number" in meta:
                                    pages.add(meta["page_number"])
                            pages_processed = len(pages) if pages else 1
                            logger.info(f"Loaded existing database collection {file_hash} with {pages_processed} pages.")
                        else:
                            db_exists = False
                            
                        # If not exist, process PDF and populate store
                        if not db_exists:
                            chunks = process_pdf(file_bytes, uploaded_file.name)
                            db = get_vector_store(file_hash, chunks)
                            pages_processed = len(set(c.metadata.get("page_number", 0) for c in chunks))
                            logger.info(f"Processed new document: {uploaded_file.name} into {pages_processed} pages.")
                            
                        # Save status to session state
                        st.session_state["collection_name"] = file_hash
                        st.session_state["active_file_name"] = uploaded_file.name
                        st.session_state["active_file_size"] = size_str
                        st.session_state["pages_processed"] = pages_processed
                        st.session_state["db"] = db
                        
                        st.success(f"🎉 Successfully processed {pages_processed} pages!")
                        st.rerun()
                        
                    except ValueError as val_err:
                        st.error(f"⚠️ Document validation failed: {str(val_err)}")
                        logger.warning(f"Validation failure on {uploaded_file.name}: {str(val_err)}")
                    except Exception as e:
                        st.error(
                            "❌ This PDF could not be read. It may be corrupt, password-protected, "
                            "or contains scanned images without OCR text selectable elements."
                        )
                        logger.error(f"Error handling PDF parsing/embedding: {str(e)}", exc_info=True)
                        
    # 2. Document details and clear button (if loaded)
    else:
        st.markdown("### 📄 Active Document")
        st.markdown(
            f"""
            <div class="meta-card">
                <div class="meta-item">
                    <span class="meta-label">File:</span>
                    <span class="meta-value" style="word-break: break-all;">{st.session_state["active_file_name"]}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Size:</span>
                    <span class="meta-value">{st.session_state["active_file_size"]}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Pages:</span>
                    <span class="meta-value">{st.session_state["pages_processed"]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Reset button inside the sidebar
        if st.button("🔄 Clear / New PDF", type="primary", use_container_width=True):
            reset_session()
            st.rerun()

# Main Application Panel
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">DocuMind / <span>PDF Chatbot</span></h1>
        <p class="app-subtitle">RAG-powered conversational engine to explore, analyze, and query your PDF documents with page-level citations.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# API key validation
if not config.GEMINI_API_KEY:
    st.error("❌ Google Gemini API Key is missing. Please define the GEMINI_API_KEY variable in your local `.env` file to start the chatbot.")
    st.stop()


# 1. Welcome Screen (if no active collection is loaded in the main panel)
if not st.session_state["collection_name"]:
    st.info("👈 Please upload a PDF document in the sidebar to begin chatting.")

# 2. Chat Interface Phase (Show only when a document is active)
else:
    # Full-width chat banner in main panel
    st.markdown(
        f"""
        <div class="chat-banner" style="margin-bottom: 1.5rem; display: flex; align-items: center; gap: 8px;">
            <span>💬 Chatting with:</span>
            <strong style="color: #ffffff; word-break: break-all;">{st.session_state["active_file_name"]}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display Chat History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Render page citations and model badges for assistant responses
            if msg["role"] == "assistant":
                source_pages = msg.get("source_pages", [])
                model_used = msg.get("model_used", "")
                
                # Check if we have any metadata to display
                if source_pages or model_used:
                    citation_html = '<div class="citation-container">'
                    
                    if source_pages:
                        page_chips = " ".join([f'<span class="citation-chip">Page {p}</span>' for p in source_pages])
                        citation_html += f'<span>Sources: {page_chips}</span>'
                        
                    if model_used:
                        citation_html += f'<span class="model-badge">Model: {model_used}</span>'
                        
                    citation_html += '</div>'
                    st.markdown(citation_html, unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input("Ask a question about this document...")
    
    if user_query:
        # 1. Display User Message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # 2. Run Query through RAG pipeline in streaming mode
        with st.chat_message("assistant"):
            try:
                db = st.session_state["db"]
                chat_history = st.session_state["messages"][:-1]  # exclude the current message we just appended
                
                # Shared metadata container (passed by reference to generator)
                meta = {"source_pages": [], "model_used": ""}
                
                # Show immediate status placeholder
                status_placeholder = st.empty()
                status_placeholder.markdown("🔍 *DocuMind is retrieving facts and referencing pages...*")
                
                # Stream the response using st.write_stream
                stream_generator = query_rag_chain_stream(user_query, chat_history, db, meta)
                
                def stream_wrapper():
                    cleared = False
                    for chunk in stream_generator:
                        if not cleared:
                            status_placeholder.empty()
                            cleared = True
                        yield chunk
                    if not cleared:
                        status_placeholder.empty()
                        
                answer = st.write_stream(stream_wrapper())

                
                source_pages = meta["source_pages"]
                model_used = meta["model_used"]
                
                # Display citations and model badge below completed stream
                if source_pages or model_used:
                    citation_html = '<div class="citation-container">'
                    if source_pages:
                        page_chips = " ".join([f'<span class="citation-chip">Page {p}</span>' for p in source_pages])
                        citation_html += f'<span>Sources: {page_chips}</span>'
                    if model_used:
                        citation_html += f'<span class="model-badge">Model: {model_used}</span>'
                    citation_html += '</div>'
                    st.markdown(citation_html, unsafe_allow_html=True)
                
                # Save Assistant Response
                assistant_msg = {
                    "role": "assistant",
                    "content": answer,
                    "source_pages": source_pages,
                    "model_used": model_used
                }
                st.session_state["messages"].append(assistant_msg)
                
                st.rerun()
                
            except Exception as e:
                friendly_error = (
                    "Sorry, I encountered an error while retrieving your answer. "
                    "This could be due to a temporary network issue or an invalid API Key. "
                    "Please verify your connection and key, then try again."
                )
                st.error(friendly_error)
                logger.error(f"Error executing Chat RAG Chain query: {str(e)}", exc_info=True)

