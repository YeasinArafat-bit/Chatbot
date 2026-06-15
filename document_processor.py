import patch_protobuf
import os
import tempfile

import time
import hashlib
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

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

def process_pdf(file_bytes: bytes, file_name: str) -> List[Document]:
    """Loads PDF from bytes, extracts text per page, validates the content,
    splits it into chunks, and returns the list of chunks with 1-indexed page metadata.
    """
    logger.info(f"Starting processing of file: {file_name}")
    
    # Save uploaded bytes to a temporary file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
        
    try:
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        logger.info(f"Loaded PDF with {len(docs)} pages.")
        
        # Validate that the PDF is not scanned/empty (text extraction validation)
        total_text = "".join([doc.page_content for doc in docs])
        logger.info(f"Total extracted text length: {len(total_text)} characters.")
        
        if len(total_text.strip()) < 100:
            raise ValueError(
                "This PDF appears to be a scanned image or contains no selectable text. "
                "The chatbot cannot read scanned/image-only PDFs without OCR."
            )
            
        # Text Splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )
        chunks = splitter.split_documents(docs)
        
        # Ensure 1-indexed page number metadata is present
        for chunk in chunks:
            # PyPDFLoader usually stores 0-indexed page in chunk.metadata['page']
            page_idx = chunk.metadata.get("page", 0)
            chunk.metadata["page_number"] = page_idx + 1
            
        logger.info(f"Successfully split PDF into {len(chunks)} chunks.")
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
    def __init__(self, base_embeddings, batch_size=20, sleep_between_batches=3.0):
        self.base_embeddings = base_embeddings
        self.batch_size = batch_size
        self.sleep_between_batches = sleep_between_batches
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        total_texts = len(texts)
        total_batches = -(-total_texts // self.batch_size) # ceil division
        
        logger.info(f"RateLimitedEmbeddings: Starting embedding of {total_texts} chunks in {total_batches} batches.")
        
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.info(f"RateLimitedEmbeddings: Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            retries = 5
            backoff = 10.0
            while retries > 0:
                try:
                    batch_embeddings = self.base_embeddings.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        logger.warning(
                            f"RateLimitedEmbeddings: Rate limit hit (429/RESOURCE_EXHAUSTED) on batch {batch_num}. "
                            f"Waiting {backoff} seconds before retry. {retries} retries remaining."
                        )
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

def get_vector_store(collection_name: str, chunks: List[Document] = None) -> Chroma:
    """Retrieves or creates a Chroma vector store for the given collection name.
    If chunks are provided and the store is empty, it will add the chunks to the store.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        
    logger.info(f"Accessing vector store for collection: {collection_name}")
    
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GEMINI_API_KEY
    )
    embeddings = RateLimitedEmbeddings(base_embeddings)
    
    # Initialize Chroma db
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.VECTOR_STORE_PATH
    )
    
    # Check if documents already exist in the collection
    try:
        existing_count = db._collection.count()
    except Exception as e:
        logger.warning(f"Failed to check existing collection count: {str(e)}")
        existing_count = 0
        
    if existing_count > 0:
        logger.info(f"Collection {collection_name} already populated with {existing_count} chunks. Skipping embedding.")
        return db
        
    if chunks:
        logger.info(f"Embedding {len(chunks)} chunks into collection {collection_name}...")
        db.add_documents(chunks)
        logger.info("Embedding and storage complete.")
    else:
        logger.warning(f"Collection {collection_name} is empty and no chunks were provided to populate it.")
        
    return db
