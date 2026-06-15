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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# File Upload Limit
try:
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
except ValueError:
    MAX_FILE_SIZE_MB = 10

# Text Splitter Settings
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
except ValueError:
    CHUNK_SIZE = 1000

try:
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
except ValueError:
    CHUNK_OVERLAP = 150

# Vector Store Path
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store")
