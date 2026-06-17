# OS/ENVIRONMENT
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import traceback

# DB
import db_logger
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

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

DATABASE_URL = os.getenv("DATABASE_URL")

########################################################################
# FASTAPI APP, STARTUP AND SHUTDOWN
########################################################################

# use lifespan func as a constructor to prep environment for fasAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):

    # prep OpanAI client and Supabase
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=10)

    # register vector type ONCE
    with pool.connection() as conn:
        register_vector(conn)

    # initialize response db
    db_logger.init_db()

    # attach clients to app instance
    app.state.pool = pool
    app.state.openai_client = openai_client

    print("[startup] OpenAI and Supabase Loaded.")

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

# retrieve n relevant chunks of text with their metadata from an embedded question
def retrieve(pool: ConnectionPool, vector: list[float], n: int) -> tuple[list[str], list[dict]]:
    # establish connection with psycopg and enable pgvector
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content, doc, page
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """, (vector, n))
            rows = cur.fetchall()
    chunks = [row[0] for row in rows]
    metas = [{"source": row[1], "page": row[2]} for row in rows]
    return chunks, metas

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
        # check supabase connection
        with app.state.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                count = cur.fetchone()[0]
        return {"status": "ok", "chunk_count": count}
    
    # if error, chromaDB is somehow broken or not available
    except Exception as e:
        # exception 503- unable to handle request
        raise HTTPException(status_code=503, detail=f"Supabase unavailable: {str(e)}")
    
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # start clocking time
    t0 = time.perf_counter()

    try:
        # turn question into vector using embedding
        vector = embed(app.state.openai_client, req.question)
        # get relevant chunks and sources based on embed vector
        chunks, metas = retrieve(app.state.pool, vector, req.n_results)
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
        traceback.print_exc()
        #error code 500, generic catch all
        raise HTTPException(status_code=500, detail=str(e))
