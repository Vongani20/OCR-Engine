# config.py
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input" / "invoices"
PROCESSED_DIR = BASE_DIR / "input" / "processed"
OUTPUT_DATA_DIR = BASE_DIR / "output" / "extracted_data"
OUTPUT_REPORT_DIR = BASE_DIR / "output" / "combined_report"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [INPUT_DIR, PROCESSED_DIR, OUTPUT_DATA_DIR, OUTPUT_REPORT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Gemini configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBy-LeG71c0HLkC8xr6Jp6-uKpQN3R1VL8")
MODEL_NAME = "gemini-1.5-pro"  # or "gemini-1.5-flash" for speed
TEMPERATURE = 0.1

# Supported file types
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf'}

# Processing settings
PDF_DPI = 200
MAX_WORKERS = 4