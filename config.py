import patch_protobuf
import os
from dotenv import load_dotenv


# Load .env file if it exists
load_dotenv()

# Gemini Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-flash-latest")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")

# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower()
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"))
EMBEDDING_MODEL = GOOGLE_EMBEDDING_MODEL
try:
    SHORT_PDF_THRESHOLD_PAGES = int(os.getenv("SHORT_PDF_THRESHOLD_PAGES", "10"))
except ValueError:
    SHORT_PDF_THRESHOLD_PAGES = 10

# File Upload Limit
try:
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
except ValueError:
    MAX_FILE_SIZE_MB = 10

# Text Splitter Settings
try:
    PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
except ValueError:
    PARENT_CHUNK_SIZE = 2000

try:
    PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
except ValueError:
    PARENT_CHUNK_OVERLAP = 200

try:
    CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "500"))
except ValueError:
    CHILD_CHUNK_SIZE = 500

try:
    CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
except ValueError:
    CHILD_CHUNK_OVERLAP = 50

# Vector Store Path
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")

# Embedding API settings for faster ingestion
try:
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1000"))
except ValueError:
    EMBEDDING_BATCH_SIZE = 1000

try:
    EMBEDDING_SLEEP_SECONDS = float(os.getenv("EMBEDDING_SLEEP_SECONDS", "0.2"))
except ValueError:
    EMBEDDING_SLEEP_SECONDS = 0.2
# Rate limiting for Gemini free tier API limits
ENABLE_RATE_LIMITER = os.getenv("ENABLE_RATE_LIMITER", "True").lower() in ("true", "1", "yes")

# Hybrid Retrieval Settings
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
try:
    SEMANTIC_SEARCH_WEIGHT = float(os.getenv("SEMANTIC_SEARCH_WEIGHT", "0.5"))
except ValueError:
    SEMANTIC_SEARCH_WEIGHT = 0.5

try:
    KEYWORD_SEARCH_WEIGHT = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.5"))
except ValueError:
    KEYWORD_SEARCH_WEIGHT = 0.5
