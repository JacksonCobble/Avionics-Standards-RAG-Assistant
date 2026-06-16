# OS/ENVIRONMENT
import os
from pathlib import Path
from dotenv import load_dotenv

# DB
import chromadb

# API/SERVER
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from openai import OpenAI

# DATA VALIDATION
from models import QueryRequest, QueryResponse, Source

# prep environment
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# use lifespan func as a constructor to prep environment for fasAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # get paths to chromaDB
    project_root = Path(__file__).parent.parent
    chroma_path  = Path(CHROMA_PATH)
    if not chroma_path.is_absolute():
        chroma_path = project_root / chroma_path

    # prep chroma and OpanAI client
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection    = chroma_client.get_collection(COLLECTION_NAME)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # make sure chroma DB loaded
    count = collection.count()
    print(f"[startup] ChromaDB ready — {count} chunks in '{COLLECTION_NAME}'")

    # attach clients to app instance
    app.state.collection    = collection
    app.state.openai_client = openai_client

    # everything under the yield will run on shutdown
    yield

    print("[shutdown] Closing down.")

app = FastAPI(title="Avionics RAG API", lifespan=lifespan)

# check health of server running
@app.get("/health")
async def health():
    try:
        # check chromaDB, return informatics on it
        count = app.state.collection.count()
        return {"status": "ok", "chroma_count": count, "collection": COLLECTION_NAME}
    # if error, chromaDB is somehow broken or not available
    except Exception as e:
        # exception 503- unable to handle request
        raise HTTPException(status_code=503, detail=f"ChromaDB Unavailable: {str(e)}")