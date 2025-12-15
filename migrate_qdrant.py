"""
Qdrant Migration Script
Uploads all local Qdrant collections to Qdrant Cloud.

Usage:
    1. Set QDRANT_URL and QDRANT_API_KEY in your .env (or as env vars below)
    2. Run: python migrate_qdrant.py
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from pathlib import Path

load_dotenv()

LOCAL_QDRANT_PATH = str(Path("data_artifacts/qdrant_storage"))
CLOUD_URL = os.getenv("QDRANT_URL")
CLOUD_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTIONS = [
    "inter_iit_conversations",
    "inter_iit_knowledge_graph",
]

BATCH_SIZE = 100

def migrate():
    if not CLOUD_URL or not CLOUD_API_KEY:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in your .env file.")
        return

    print(f"Connecting to local Qdrant at: {LOCAL_QDRANT_PATH}")
    local = QdrantClient(path=LOCAL_QDRANT_PATH)

    print(f"Connecting to Qdrant Cloud at: {CLOUD_URL}")
    cloud = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY)

    for collection_name in COLLECTIONS:
        print(f"\n--- Migrating collection: {collection_name} ---")

        # Get local collection info
        info = local.get_collection(collection_name)
        vector_size = info.config.params.vectors.size
        distance = info.config.params.vectors.distance
        total_points = info.points_count

        print(f"  Points: {total_points}, Vector size: {vector_size}, Distance: {distance}")

        # Create collection on cloud if it doesn't exist
        existing = [c.name for c in cloud.get_collections().collections]
        if collection_name in existing:
            print(f"  Collection already exists on cloud, skipping creation.")
        else:
            cloud.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            print(f"  Created collection on cloud.")

        # Scroll and upload in batches
        offset = None
        total_uploaded = 0

        while True:
            result, next_offset = local.scroll(
                collection_name=collection_name,
                limit=BATCH_SIZE,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )

            if not result:
                break

            ids = [p.id for p in result]
            vectors = [p.vector for p in result]
            payloads = [p.payload for p in result]

            cloud.upsert(
                collection_name=collection_name,
                points=[
                    {"id": pid, "vector": vec, "payload": pay}
                    for pid, vec, pay in zip(ids, vectors, payloads)
                ],
            )

            total_uploaded += len(result)
            print(f"  Uploaded {total_uploaded}/{total_points} points...", end="\r")

            if next_offset is None:
                break
            offset = next_offset

        print(f"\n  Done! Uploaded {total_uploaded} points to '{collection_name}'.")

    print("\n✅ Migration complete! All collections are now on Qdrant Cloud.")

if __name__ == "__main__":
    migrate()
