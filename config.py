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
