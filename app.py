import patch_protobuf
import os
import streamlit as st

import config
from logger import get_logger
# Lazy load modules inside callbacks for instant initial page loading

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

/* Hide upload icon / plus sign in file uploader */
[data-testid="stFileUploader"] svg, [data-testid="stFileUploaderIcon"] {
    display: none !important;
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
        
        # Disable uploader if a file is already being processed to prevent double-upload clicks
        is_disabled = False
        if "pdf_uploader" in st.session_state and st.session_state["pdf_uploader"] is not None:
            is_disabled = True
            
        uploaded_file = st.file_uploader(
            "Choose a PDF file to begin",
            type=["pdf"],
            key="pdf_uploader",
            disabled=is_disabled,
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
                from document_processor import (
                    calculate_file_hash, 
                    get_vector_store, 
                    process_pdf,
                    count_pdf_pages,
                    get_stored_provider
                )
                file_bytes = uploaded_file.read()
                file_hash = calculate_file_hash(file_bytes)
                
                # Formatted file size string
                if file_size_mb < 0.1:
                    size_str = f"{uploaded_file.size / 1024:.2f} KB"
                else:
                    size_str = f"{file_size_mb:.2f} MB"
                    
                logger.info(f"File uploaded successfully: {uploaded_file.name} ({size_str}), Hash: {file_hash}")
                # Render the custom CSS loading line immediately!
                loading_placeholder = st.sidebar.empty()
                loading_html = """
                <div class="loading-line-container">
                    <div class="loading-line"></div>
                </div>
                <style>
                .loading-line-container {
                    width: 100%;
                    height: 4px;
                    background-color: #334155;
                    border-radius: 2px;
                    overflow: hidden;
                    margin-top: 15px;
                    margin-bottom: 15px;
                }
                .loading-line {
                    height: 100%;
                    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
                    box-shadow: 0 0 8px #a855f7;
                    width: 0%;
                    animation: crawl 45s cubic-bezier(0.1, 0.8, 0.1, 1.0) forwards, pulse 1.5s infinite alternate;
                }
                @keyframes crawl {
                    0% { width: 0%; }
                    10% { width: 35%; }
                    30% { width: 65%; }
                    60% { width: 88%; }
                    100% { width: 98%; }
                }
                @keyframes pulse {
                    0% { opacity: 0.7; }
                    100% { opacity: 1; }
                }
                </style>
                <div style="font-size: 0.85rem; color: #94a3b8; text-align: center; margin-top: -10px;">
                    ⚡ Loading document...
                </div>
                """
                loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
                import time
                time.sleep(0.1) # Yield control to Streamlit event loop to render the loader in the browser
                
                try:
                    # Determine provider config to avoid rate limits
                    stored_provider = get_stored_provider(file_hash)
                    if stored_provider is not None:
                        provider = stored_provider
                    else:
                        provider = config.EMBEDDING_PROVIDER
                        logger.info(f"Using configured EMBEDDING_PROVIDER: {provider}")
                    
                    # 1. Fast check if database already exists (silent check)
                    db_exists = False
                    existing_db = get_vector_store(file_hash, chunks=None, provider=provider, progress_callback=None)
                    
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
                        logger.info(f"Loaded existing database collection {file_hash} ({provider}) with {pages_processed} pages.")
                    else:
                        db_exists = False
                        
                    # 2. If not exist, process PDF and rebuild store
                    if not db_exists:
                        chunks = process_pdf(file_bytes, uploaded_file.name, progress_callback=None)
                        db = get_vector_store(file_hash, chunks, provider=provider, progress_callback=None)
                        pages_processed = len(set(c.metadata.get("page_number", 0) for c in chunks))
                        logger.info(f"Processed new document: {uploaded_file.name} into {pages_processed} pages using {provider} provider.")
                    else:
                        db = existing_db
                        
                    # Save status to session state
                    st.session_state["collection_name"] = file_hash
                    st.session_state["active_file_name"] = uploaded_file.name
                    st.session_state["active_file_size"] = size_str
                    st.session_state["pages_processed"] = pages_processed
                    st.session_state["db"] = db
                    
                    # Clear loading animation
                    loading_placeholder.empty()
                    st.rerun()
                        
                except ValueError as val_err:
                    loading_placeholder.empty()
                    st.error(f"⚠️ Document validation failed: {str(val_err)}")
                    logger.warning(f"Validation failure on {uploaded_file.name}: {str(val_err)}")
                except Exception as e:
                    loading_placeholder.empty()
                    st.error(
                        f"❌ This PDF could not be read (Error: {str(e)}).\n\n"
                        "It may be corrupt, password-protected, or contain scanned images without OCR text selectable elements."
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
                
                # Check if we have page citations or model info to display
                if source_pages or model_used:
                    citation_html = '<div class="citation-container">'
                    if source_pages:
                        page_chips = " ".join([f'<span class="citation-chip">Page {p}</span>' for p in source_pages])
                        citation_html += f'<span>Sources: {page_chips}</span>'
                    if model_used:
                        display_model = model_used.split("/")[-1]
                        citation_html += f'<span class="model-badge">🤖 {display_model}</span>'
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
                
                # Show a simple, clean spinner while searching and preparing stream
                with st.spinner("Processing..."):
                    from rag_chain import query_rag_chain_stream
                    stream_generator = query_rag_chain_stream(
                        user_query, chat_history, db, meta, status_callback=None
                    )
                    
                    # Consume first chunk to trigger retrieval steps under the spinner
                    try:
                        first_chunk = next(stream_generator)
                        has_chunks = True
                    except StopIteration:
                        first_chunk = ""
                        has_chunks = False
                
                # Stream response outside of the spinner to print ChatGPT letter-by-letter style
                if has_chunks:
                    def stream_wrapper():
                        import time
                        # Yield the first chunk letter-by-letter smoothly
                        for char in first_chunk:
                            yield char
                            time.sleep(0.02) # standard delay for smooth letter-by-letter typing (20ms)
                        # Yield the remaining chunks letter-by-letter smoothly
                        for chunk in stream_generator:
                            for char in chunk:
                                yield char
                                time.sleep(0.02)
                                
                    answer = st.write_stream(stream_wrapper())
                else:
                    answer = "I couldn't retrieve an answer."
                    st.write(answer)

                # Post-process response to see if it failed to find answer
                not_found_indicators = [
                    "couldn't find this in the document",
                    "could not find this in the document",
                    "i couldn't find this"
                ]
                
                source_pages = []
                if any(indicator in answer.lower() for indicator in not_found_indicators):
                    pass
                else:
                    # Parse sources from LLM response
                    import re
                    sources_match = re.search(r'\[Sources:\s*(.*?)\]', answer)
                    if sources_match:
                        sources_str = sources_match.group(1)
                        # Remove the raw tag from the saved answer text
                        answer = answer.replace(sources_match.group(0), "").strip()
                        if "none" not in sources_str.lower():
                            page_nums = re.findall(r'\d+', sources_str)
                            source_pages = sorted(list(set(int(p) for p in page_nums)))
                    
                    if not source_pages:
                        # Instantly retrieve unique pages from docs as fallback
                        from rag_chain import filter_used_pages
                        source_pages = filter_used_pages(
                            answer,
                            meta.get("retrieved_docs", []),
                            meta.get("primary_model"),
                            meta.get("fallback_model"),
                            question=user_query
                        )
                
                model_used = meta["model_used"]
                
                # Display citations and model badge below completed stream
                if source_pages or model_used:
                    citation_html = '<div class="citation-container">'
                    if source_pages:
                        page_chips = " ".join([f'<span class="citation-chip">Page {p}</span>' for p in source_pages])
                        citation_html += f'<span>Sources: {page_chips}</span>'
                    if model_used:
                        display_model = model_used.split("/")[-1]
                        citation_html += f'<span class="model-badge">🤖 {display_model}</span>'
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
                err_msg = str(e).lower()
                friendly_error = "Sorry, I encountered an error while retrieving your answer."
                
                if any(x in err_msg for x in ["connection", "timeout", "dns", "newconnectionerror", "maxretryerror", "host", "socket"]):
                    friendly_error = "❌ Network Connection Issue: Could not connect to the API. Please check your internet connection and try again."
                elif any(x in err_msg for x in ["api_key_invalid", "invalid_api_key", "unauthorized", "api key is not valid", "401"]):
                    friendly_error = "🔑 Invalid API Key: Please verify that your GEMINI_API_KEY is correct in your local .env file."
                elif any(x in err_msg for x in ["resource_exhausted", "429", "quota", "rate limit"]):
                    friendly_error = "⚠️ Rate Limit Exceeded: You have reached the Gemini API quota limit. Please wait a minute and try again."
                else:
                    friendly_error = f"❌ Error: {str(e)}"
                    
                st.error(friendly_error)
                logger.error(f"Error executing Chat RAG Chain query: {str(e)}", exc_info=True)

