import logging
import sys
from pathlib import Path

# Create log directory if it doesn't exist
log_file = Path("app.log")

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the specified name."""
    return logging.getLogger(name)
