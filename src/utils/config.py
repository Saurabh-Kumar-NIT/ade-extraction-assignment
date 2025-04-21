import os
from pathlib import Path

# Project paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted" / "cadec"
TEXT_DIR = EXTRACTED_DATA_DIR / "text"
ORIGINAL_ANNOTATIONS_DIR = EXTRACTED_DATA_DIR / "original"
MEDDRA_ANNOTATIONS_DIR = EXTRACTED_DATA_DIR / "meddra"
SCT_ANNOTATIONS_DIR = EXTRACTED_DATA_DIR / "sct"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Model settings
ENTITY_EXTRACTION_MODEL = "google/flan-t5-large"
ABBREVIATION_EXPANSION_MODEL = "google/flan-t5-base"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# UMLS API settings
UMLS_API_KEY = os.getenv("UMLS_API_KEY", "")
UMLS_API_URL = "https://uts-ws.nlm.nih.gov/rest"

# Verification settings
MAX_RETRIES = 3
SIMILARITY_THRESHOLD = 0.75
SCHEMA_VALIDATION_ENABLED = True

# Output formats
EXTRACTED_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "drugs": {
            "type": "array",
            "items": {"type": "string"}
        },
        "adverse_events": {
            "type": "array",
            "items": {"type": "string"}
        },
        "symptoms": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["drugs", "adverse_events", "symptoms"]
}



