import os
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
CHROMA_PATH     = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "avionics_standards")

@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).parent.parent
    chroma_path  = Path(CHROMA_PATH)
    if not chroma_path.is_absolute():
        chroma_path = project_root / chroma_path

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection    = chroma_client.get_collection(COLLECTION_NAME)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    count = collection.count()
    print(f"[startup] ChromaDB ready — {count} chunks in '{COLLECTION_NAME}'")

    app.state.collection    = collection
    app.state.openai_client = openai_client

    yield

    print("[shutdown] Closing down.")

app = FastAPI(title="Avionics RAG API", lifespan=lifespan)