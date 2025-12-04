import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "data_artifacts"
GRAPH_PATH = ARTIFACTS_DIR / "knowledge_graph.gpickle"
QDRANT_PATH = ARTIFACTS_DIR / "qdrant_storage"

COLLECTION_NAME = "inter_iit_knowledge_graph"

MODEL_ID = "google/embeddinggemma-300m"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY not found in environment variables.")