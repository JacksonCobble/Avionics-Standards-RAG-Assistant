# OS/ENVIRONMENT
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# DB
import chromadb
import db_logger

# API/SERVER
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from openai import OpenAI

# DATA VALIDATION
from models import QueryRequest, QueryResponse, Source

########################################################################
# ENVIRONMENT SETUP
########################################################################

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


########################################################################
# FASTAPI APP, STARTUP AND SHUTDOWN
########################################################################

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

    # initialize response db
    db_logger.init_db()

    # attach clients to app instance
    app.state.collection    = collection
    app.state.openai_client = openai_client

    # everything under the yield will run on shutdown
    yield

    print("[shutdown] Closing down.")

########################################################################
# LLM FUNCTIONS
########################################################################

# embed text and return its vector
def embed(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

# retrieve n relevant chunks from chromaDB for a given vector
def retrieve(collection, vector: list[float], n: int) -> tuple[list[str], list[dict]]:
    results = collection.query(
        query_embeddings=[vector],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    # query designed to handle batches- because we are using a singular query, we can use the first element returned as there will only be 1
    return results["documents"][0], results["metadatas"][0]

# turns list of relevant chunks into a formatted block of text displaying each chunk with its relevant information
def build_context_block(chunks: list[str], metas: list[dict]) -> str:
    parts = []
    #go over each chunk and its metadata
    for i, (doc, meta) in enumerate(zip(chunks, metas), 1):
        source = meta.get("source", "unknown source")
        page = meta.get("page", "unknown page")
        # format chunk information in a block to build cohesive prompt for LLM
        parts.append(f"[Chunk {i} | Source {source} | Page: {page}]\n{doc}")
    return "\n\n---\n\n".join(parts)

# given question and context block, prompt LLM and return answer
def call_llm(client: OpenAI, question: str, context: str) -> str:
    # try to make format of prompt good looking enough to feel like its well written
    prompt = (f"You are a technical expert on avionics data bus standards.\n"
                f"Answer the question below using ONLY the provided context.\n"
                f"If the context does not contain enough information, say so — do not speculate.\n"
                f"When you cite a fact, reference its source and page like this: [SOURCE: filename, p.N].\n\n"
                f"CONTEXT: {context}\n\n"
                f"QUESTION: {question} \n\n"
                f"ANSWER:")

    # query model with prompt
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise technical assistant for avionics standards. "
                    "Answer only from provided context. "
                    "Never fabricate specifications or figures."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    # return answer
    return completion.choices[0].message.content.strip()

# remove duplicate chunk sources if we have multiple referencing the same page
def check_for_duplicate_sources(metas: list[dict]) -> list[Source]:
    seen   = set()
    result = []
    # go over each meta and save source and page one time if we dont have a copy of it
    for m in metas:
        key = (m.get("source", "unknown"), int(m.get("page", 0)))
        if key not in seen:
            seen.add(key)
            result.append(Source(doc=key[0], page=key[1]))
    # return our list with duplicates removed
    return sorted(result, key=lambda s: (s.doc, s.page))

########################################################################
# FASTAPI ENDPOINTS
########################################################################

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
    
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # start clocking time
    t0 = time.perf_counter()

    try:
        # turn question into vector using embedding
        vector = embed(app.state.openai_client, req.question)
        # get relevant chunks and sources based on embed vector
        chunks, metas = retrieve(app.state.collection, vector, req.n_results)
        # build context block for prompt
        context = build_context_block(chunks, metas)
        # build query and get answer from LLM
        llm_response = call_llm(app.state.openai_client, req.question, context)
        # remove any dupe sources from chunks
        no_dupes_sources = check_for_duplicate_sources(metas)
        #get latency of full operation
        operation_time_ms = (time.perf_counter() - t0) * 1000

        # build response object
        response =  QueryResponse(
            question = req.question,
            answer = llm_response,
            sources = no_dupes_sources,
            chunks_used = len(chunks),
            latency_ms = round(operation_time_ms, 1)
        )

        # log to response db and return
        db_logger.log_query(response)
        return response
    
    except Exception as e:
        #error code 500, generic catch all
        raise HTTPException(status_code=500, detail=str(e))
