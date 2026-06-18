import patch_protobuf
import os
import tempfile
import pickle

import time
import hashlib
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import config
from logger import get_logger

logger = get_logger(__name__)

def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculates the MD5 hash of file bytes and returns a valid Chroma collection name.
    Chroma collection names must be 3-63 chars, start/end with alphanumeric, and contain
    only alphanumeric, underscores, or hyphens.
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return f"pdf_{file_hash}"

def count_pdf_pages(file_bytes: bytes) -> int:
    """Counts the pages of a PDF from bytes quickly using pypdf.PdfReader."""
    import io
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to count PDF pages: {e}")
        return 1

def get_stored_provider(collection_name: str) -> str:
    """Retrieves the stored embedding provider from Chroma collection metadata
    without loading the embedding model itself.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=config.VECTOR_STORE_PATH)
        collection = client.get_collection(name=collection_name)
        if collection.metadata:
            # If chunking mode is not page_level or chunk_size doesn't match, force re-indexing
            if collection.metadata.get("chunking_mode") != "page_level":
                return None
            stored_chunk_size = collection.metadata.get("chunk_size")
            if stored_chunk_size != config.CHILD_CHUNK_SIZE:
                return None
            return collection.metadata.get("provider")
    except Exception:
        pass
    return None

def process_pdf(file_bytes: bytes, file_name: str, progress_callback=None) -> List[Document]:
    """Loads PDF from bytes, extracts text per page, validates the content,
    splits it into chunks, and returns the list of chunks with 1-indexed page metadata.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    logger.info(f"Starting processing of file: {file_name}")
    if progress_callback:
        progress_callback("📂 Processing document...")
    
    # Save uploaded bytes to a temporary file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
        
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        logger.info(f"Loaded PDF with {len(docs)} pages.")
        
        if progress_callback:
            progress_callback("📄 Processing document...")
            
        # Validate that the PDF is not scanned/empty (text extraction validation)
        total_text = "".join([doc.page_content for doc in docs])
        logger.info(f"Total extracted text length: {len(total_text)} characters.")
        
        if len(total_text.strip()) < 100:
            raise ValueError(
                "This PDF appears to be a scanned image or contains no selectable text. "
                "The chatbot cannot read scanned/image-only PDFs without OCR."
            )
            
        if progress_callback:
            progress_callback("⚙️ Processing document...")

        # Create the text splitter for page content splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP,
            length_function=len
        )
        
        chunks = []
        # Process page-by-page
        for doc in docs:
            page_idx = doc.metadata.get("page", 0)
            page_num = page_idx + 1
            
            # Split this page's text into sub-chunks
            page_text = doc.page_content
            sub_texts = splitter.split_text(page_text)
            
            for sub_text in sub_texts:
                # Prepend context breadcrumb directly to the chunk text
                context_breadcrumb = f"[Source: {file_name}, Page {page_num}] "
                injected_text = f"{context_breadcrumb}{sub_text}"
                
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["page_number"] = page_num
                chunk_metadata["parent_id"] = None
                chunk_metadata["parent_content"] = None
                
                chunks.append(Document(
                    page_content=injected_text,
                    metadata=chunk_metadata
                ))
                
        logger.info(f"Split PDF page-by-page into {len(chunks)} contextual flat chunks.")
        if progress_callback:
            progress_callback("✓ Almost there...")
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise e
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

class RateLimitedEmbeddings(Embeddings):
    """A wrapper for LangChain Embeddings that batches requests and adds retry logic 
    with exponential backoff when encountering rate limits (HTTP 429 / RESOURCE_EXHAUSTED).
    """
    def __init__(self, base_embeddings, batch_size=None, sleep_between_batches=None, progress_callback=None):
        self.base_embeddings = base_embeddings
        self.batch_size = batch_size if batch_size is not None else config.EMBEDDING_BATCH_SIZE
        self.sleep_between_batches = sleep_between_batches if sleep_between_batches is not None else config.EMBEDDING_SLEEP_SECONDS
        self.progress_callback = progress_callback
        self.request_history = [] # list of tuples: (timestamp, char_count)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        total_texts = len(texts)
        total_batches = -(-total_texts // self.batch_size) # ceil division
        
        logger.info(f"RateLimitedEmbeddings: Starting embedding of {total_texts} chunks in {total_batches} batches.")
        
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            # Apply sliding window rate limiter if enabled
            if config.ENABLE_RATE_LIMITER:
                while True:
                    now = time.time()
                    # Keep history of only the last 60 seconds
                    self.request_history = [r for r in self.request_history if now - r[0] < 60.0]
                    
                    recent_requests = len(self.request_history)
                    
                    # Gemini Free Tier limit: max 15 RPM (requests per minute).
                    if recent_requests >= 14:
                        if self.request_history:
                            oldest_time = self.request_history[0][0]
                            wait_time = 60.0 - (now - oldest_time) + 0.5
                            if wait_time > 0:
                                msg = f"Rate limit protection: pausing for {wait_time:.1f}s to stay under free-tier API limits..."
                                logger.info(msg)
                                if self.progress_callback:
                                    self.progress_callback(msg)
                                time.sleep(wait_time)
                        else:
                            time.sleep(5)
                            break
                    else:
                        break
            
            msg = f"Generating embeddings: batch {batch_num}/{total_batches} ({len(batch)} chunks)..."
            logger.info(f"RateLimitedEmbeddings: {msg}")
            if self.progress_callback:
                self.progress_callback(f"🧠 Processing document (batch {batch_num}/{total_batches})...")
            
            retries = 5
            backoff = 10.0
            while retries > 0:
                try:
                    batch_embeddings = self.base_embeddings.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    # Record successful request in history
                    if config.ENABLE_RATE_LIMITER:
                        self.request_history.append((time.time(), sum(len(text) for text in batch)))
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        warn_msg = (
                            f"Rate limit hit (429/RESOURCE_EXHAUSTED) on batch {batch_num}. "
                            f"Waiting {backoff} seconds before retry. {retries} retries remaining."
                        )
                        logger.warning(f"RateLimitedEmbeddings: {warn_msg}")
                        if self.progress_callback:
                            self.progress_callback(f"⚠️ Rate limit hit. Pausing for {backoff:.1f}s...")
                        time.sleep(backoff)
                        retries -= 1
                        backoff *= 2 # Exponential backoff
                    else:
                        logger.error(f"RateLimitedEmbeddings: Fatal error during embedding batch {batch_num}: {err_msg}")
                        raise e
            else:
                raise RuntimeError(
                    f"RateLimitedEmbeddings: Failed to generate embeddings for batch {batch_num}. "
                    "Rate limit quota exhausted after 5 retries."
                )
                
            # Sleep between batches to avoid hitting the rate limit
            if i + self.batch_size < total_texts:
                time.sleep(self.sleep_between_batches)
                
        logger.info("RateLimitedEmbeddings: Successfully embedded all chunks.")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.base_embeddings.embed_query(text)

import streamlit as st

@st.cache_resource(show_spinner=False)
def _load_huggingface_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info(f"Loading local embedding model ({model_name}) into cache...")
    return HuggingFaceEmbeddings(
        model_name=model_name
    )

def get_embedding_model(provider: str = None, progress_callback=None) -> Embeddings:
    """Returns the correct embedding model based on the specified provider (defaults to config.EMBEDDING_PROVIDER)."""
    if provider is None:
        provider = config.EMBEDDING_PROVIDER
        
    provider = provider.lower()
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        logger.info(
            f"Using GOOGLE embeddings (API) — Model: {config.GOOGLE_EMBEDDING_MODEL}"
        )
        base_embeddings = GoogleGenerativeAIEmbeddings(
            model=config.GOOGLE_EMBEDDING_MODEL,
            google_api_key=config.GEMINI_API_KEY
        )
        return RateLimitedEmbeddings(base_embeddings, progress_callback=progress_callback)
    else:
        # Default to local
        logger.info(
            f"Using LOCAL embeddings (sentence-transformers) — Model: {config.LOCAL_EMBEDDING_MODEL}"
        )
        if progress_callback:
            progress_callback("🧠 Initializing search engine (loading local embedding model, this may take a moment)...")
        embeddings = _load_huggingface_embeddings(config.LOCAL_EMBEDDING_MODEL)
        return embeddings

def get_vector_store(collection_name: str, chunks: List[Document] = None, provider: str = None, progress_callback=None) -> "Chroma":
    """Retrieves or creates a Chroma vector store for the given collection name using the specified provider.
    If provider is not specified, tries to read the stored provider from metadata first.
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
        
    logger.info(f"Accessing vector store for collection: {collection_name}")
    if progress_callback:
        progress_callback("🔑 Preparing search indexes...")
    
    # Check if a provider was already stored for this collection in metadata
    if provider is None:
        provider = get_stored_provider(collection_name)
        
    # If still not found, fallback to config
    if provider is None:
        provider = config.EMBEDDING_PROVIDER
        
    provider = provider.lower()
    embeddings = get_embedding_model(provider=provider, progress_callback=progress_callback)
    
    # Initialize Chroma db
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.VECTOR_STORE_PATH,
        collection_metadata={"provider": provider, "chunking_mode": "page_level", "chunk_size": config.CHILD_CHUNK_SIZE}
    )
    
    # Check if documents already exist in the collection
    try:
        existing_count = db._collection.count()
    except Exception as e:
        logger.warning(f"Failed to check existing collection count: {str(e)}")
        existing_count = 0
        
    if existing_count > 0:
        # Check stored metadata parameters to verify compatibility
        stored_metadata = db._collection.metadata
        stored_provider = None
        stored_chunking_mode = None
        stored_chunk_size = None
        if stored_metadata:
            stored_provider = stored_metadata.get("provider")
            stored_chunking_mode = stored_metadata.get("chunking_mode")
            stored_chunk_size = stored_metadata.get("chunk_size")
            
        if (stored_provider and stored_provider != provider) or (stored_chunking_mode != "page_level") or (stored_chunk_size != config.CHILD_CHUNK_SIZE):
            logger.warning(
                f"⚠️ WARNING: Incompatible database detected for collection '{collection_name}'. "
                f"Stored Provider: '{stored_provider}', Chunking: '{stored_chunking_mode}', Chunk Size: '{stored_chunk_size}'. "
                "Re-indexing collection..."
            )
            if progress_callback:
                progress_callback("⚠️ Upgrading indexing format. Re-indexing...")
            
            # Delete incompatible collection
            db.delete_collection()
            
            # Recreate Chroma db with current configuration
            db = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=config.VECTOR_STORE_PATH,
                collection_metadata={"provider": provider, "chunking_mode": "page_level", "chunk_size": config.CHILD_CHUNK_SIZE}
            )
            existing_count = 0
        else:
            logger.info(f"Collection {collection_name} already populated with {existing_count} chunks. Skipping embedding.")
            if progress_callback:
                progress_callback("✓ Loading cached index...")
            check_and_build_bm25_from_db(collection_name, db)
            if progress_callback:
                progress_callback("✓ Almost there...")
            return db
        
    if chunks:
        logger.info(f"Embedding {len(chunks)} chunks into collection {collection_name}...")
        if progress_callback:
            progress_callback("🧠 Indexing document content...")
        db.add_documents(chunks)
        logger.info("Embedding and storage complete.")
        if progress_callback:
            progress_callback("✓ Neural indexing completed.")
        build_and_persist_bm25(collection_name, chunks)
        if progress_callback:
            progress_callback("✓ Almost there...")
    else:
        logger.warning(f"Collection {collection_name} is empty and no chunks were provided to populate it.")
        
    return db

def build_and_persist_bm25(collection_name: str, chunks: List[Document]):
    """Builds a BM25Retriever from chunks and serializes it using pickle."""
    try:
        from langchain_community.retrievers import BM25Retriever
        logger.info(f"Building BM25 index for collection {collection_name} with {len(chunks)} chunks...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        
        # Ensure persist directory exists
        os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
        bm25_path = os.path.join(config.VECTOR_STORE_PATH, f"bm25_{collection_name}.pkl")
        
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        logger.info(f"Successfully persisted BM25 index to {bm25_path}")
    except Exception as e:
        logger.error(f"Failed to build/persist BM25 index: {str(e)}", exc_info=True)

def check_and_build_bm25_from_db(collection_name: str, db: "Chroma"):
    """Checks if the BM25 index is cached. If not, retrieves all docs from Chroma and builds it."""
    bm25_path = os.path.join(config.VECTOR_STORE_PATH, f"bm25_{collection_name}.pkl")
    if os.path.exists(bm25_path):
        logger.info(f"BM25 index for {collection_name} is already cached.")
        return
        
    logger.info(f"BM25 index not found at {bm25_path}. Reconstructing from Chroma database...")
    try:
        results = db.get()
        ids = results.get("ids", [])
        if not ids:
            logger.warning(f"No documents found in Chroma collection {collection_name} to build BM25.")
            return
            
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        
        chunks = []
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_text = documents[i] if i < len(documents) else ""
            chunks.append(Document(page_content=doc_text, metadata=meta))
            
        build_and_persist_bm25(collection_name, chunks)
    except Exception as e:
        logger.error(f"Failed to reconstruct BM25 from Chroma: {str(e)}", exc_info=True)
