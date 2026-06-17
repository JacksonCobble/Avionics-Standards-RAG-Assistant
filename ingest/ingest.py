# OS/ENV
import os, sys
import time
from dotenv import load_dotenv

# FILES
import hashlib
from pathlib import Path

# AI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# DB
import psycopg
from pgvector.psycopg import register_vector


########################################################################
# CONFIG
########################################################################

load_dotenv()

DOCS_DIR = Path("docs").resolve() 

CHUNK_SIZE = 1500 # size of each chunk our docs will be split into
CHUNK_OVERLAP = 150 # how many characters of overlap between chunks (to maintain context)
INSERT_BATCH_SIZE = 100 # rows per executemany call

EMBED_MODEL = os.getenv("EMBED_MODEL")
EMBED_BATCH_SIZE = 100 # chunks per OpenAI embed call


# makes hashes of files, used to skip re ingesting docs that havent changed
def hash_file(path: Path) -> str:
    hash = hashlib.sha256()
    # make sure to resolve file to get absolute path, otherwise we can get different hashes for the same file if we run the script from different directories
    with open(path.resolve(), "rb") as f:
        # Read and update hash string value in blocks of 8K in order to handle large files efficiently
        for byte_block in iter(lambda: f.read(8192), b""):
            hash.update(byte_block)
    return hash.hexdigest()

########################################################################
# SUPABASE HELPERS
########################################################################

# load hash from supabase
def load_ingested_hashes(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT file_hash FROM ingested_files")
        return {row[0] for row in cur.fetchall()}

# save a completed hash to DB
def save_ingested_hash(conn: psycopg.Connection, file_hash: str, filename: str, chunk_count: int) -> None:
    with conn.cursor() as cur:
        cur.execute("""
                    INSERT INTO ingested_files (file_hash, filename, chunk_count)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (file_hash) DO NOTHING
                    """, (file_hash, filename, chunk_count)
                    )
    conn.commit()
    return

########################################################################
# CHUNKING
########################################################################

def load_chunk_new_files(pdf_files: list[Path], ingested: set[str]) -> tuple[list[dict], dict[str,str]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", " ", ""])
    
    new_chunks: list[dict] = [] # hint: list of {content, doc, page}
    new_hashes: dict[str, str] = {}  # hint: dict of {filename, hash}  

    # go over each file
    for pdf in pdf_files:
        #get hash, and if it exists skip over it
        file_hash = hash_file(pdf)

        if file_hash in ingested:
            print(f" - Skipping {pdf.name} (already ingested)")
            continue

        # turn document into a bunch of extracted pages
        print(f" - Loading {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        # sign each page to know which pdf the content came from
        for page in pages:
            page.metadata["source"] = pdf.name

        # split docs and build chunk to add to new_chunks
        split_docs = splitter.split_documents(pages)
 
        for chunk in split_docs:
            new_chunks.append({
                "content": chunk.page_content,
                "doc":     pdf.name,
                "page":    int(chunk.metadata.get("page", 0)) + 1,
            })
 
        new_hashes[pdf.name] = file_hash
 
    return new_chunks, new_hashes

########################################################################
# EMBEDDING
########################################################################

def embed_chunks(chunks: list[dict], client: OpenAI) -> list[dict]:

    total = len(chunks)
    # slice chunks list into groups of 100
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        # send batch to be embedded
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[c["content"] for c in batch],
        )
        # throw embedding value into chunk information for each chunk in batch
        for chunk, obj in zip(batch, response.data):
            chunk["embedding"] = obj.embedding
 
        end = min(start + EMBED_BATCH_SIZE, total)
        # sanity print
        print(f"   - Embedded chunks {start + 1} - {end} / {total}")

        # add a wait incase i hit a rate limit
        if end < total:
            time.sleep(1)
 
    return chunks

########################################################################
# DB INSERTION
########################################################################

def insert_chunks(chunks: list[dict], conn: psycopg.Connection) -> None:

    # add the pgvector extension type "vector" to psycopg connection
    register_vector(conn)
    
    sql = """
        INSERT INTO documents (content, embedding, doc, page)
        VALUES (%s, %s, %s, %s)
    """
 
    total = len(chunks)
    # slice chunks into groups of 100
    for start in range(0, total, INSERT_BATCH_SIZE):
        batch = chunks[start : start + INSERT_BATCH_SIZE]
        # build db rows for a batch
        rows = [(c["content"], c["embedding"], c["doc"], c["page"]) for c in batch]
        with conn.cursor() as cur:
            # sends the same command to apply to all rows
            cur.executemany(sql, rows)
        conn.commit()
 
        end = min(start + INSERT_BATCH_SIZE, total)
        # sanity print
        print(f"   - Inserted rows {start + 1}-{end} / {total}")

########################################################################
# MAIN 
########################################################################

if __name__ == "__main__":
    
    # load from dotenv
    database_url = os.environ.get("DATABASE_URL")
    openai_key   = os.environ.get("OPENAI_API_KEY")
    if not database_url:
        raise EnvironmentError("DATABASE URL not in .env")
    if not openai_key:
        raise EnvironmentError("OPENAI KEY not in .env")
 
    client = OpenAI(api_key=openai_key)
 
    # check for PDFs in DOCS_DIR
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")
 
    print("-" * 50)
    print("AVIONICS RAG INGEST SCRIPT")
    print("-" * 50)
 
    with psycopg.connect(database_url) as conn:
 
        # 1. Check ledger
        print("[1/4] Checking ingested hashes ledger...")
        ingested = load_ingested_hashes(conn)
        print(f"   - {len(ingested)} file(s) already ingested")
 
        # 2. Load and chunk
        print("[2/4] Loading and splitting documents...")
        chunks, new_hashes = load_chunk_new_files(pdf_files, ingested)
        print(f"   - New files: {len(new_hashes)}")
        print(f"   - New chunks: {len(chunks)}")
 
        if not chunks:
            print("   - Nothing new to ingest. Exiting.")
            sys.exit(0)
 
        # 3. Embed
        print("[3/4] Creating embeddings...")
        print(f"   - Model: {EMBED_MODEL}")
        chunks = embed_chunks(chunks, client)
 
        # 4. Insert into pgvector + record hashes
        print("[4/4] Inserting into pgvector and updating ledger...")
 
        # Per-file chunk counts so we can record them accurately
        counts_by_file: dict[str, int] = {}
        for c in chunks:
            counts_by_file[c["doc"]] = counts_by_file.get(c["doc"], 0) + 1
 
        insert_chunks(chunks, conn)
 
        for filename, file_hash in new_hashes.items():
            save_ingested_hash(conn, file_hash, filename, counts_by_file[filename])
            print(f"   - Ledger updated: {filename} ({counts_by_file[filename]} chunks)")
 
    print(f"\nDone! Ingested {len(new_hashes)} new file(s), {len(chunks)} total chunks.")
