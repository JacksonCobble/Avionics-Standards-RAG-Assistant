import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# CONFIG
DOCS_DIR = Path("docs").resolve() 
CHROMA_DIR = Path(os.getenv("CHROMA_DIR")).resolve()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHUNK_SIZE = 1500 # size of each chunk our docs will be split into
CHUNK_OVERLAP = 150 # how many characters of overlap between chunks (to maintain context)
EMBED_MODEL = os.getenv("EMBED_MODEL")

# makes hashes of files, used to skip re ingesting docs that havent changed
def hash_file(path: Path) -> str:
    hash = hashlib.sha256()
    # make sure to resolve file to get absolute path, otherwise we can get different hashes for the same file if we run the script from different directories
    with open(path.resolve(), "rb") as f:
        # Read and update hash string value in blocks of 8K in order to handle large files efficiently
        for byte_block in iter(lambda: f.read(8192), b""):
            hash.update(byte_block)
    return hash.hexdigest()

# reads ledger file of ingested document hashes and returns a set of those hashes for quick lookup
def load_ingested_hashes() -> set[str]:
    ledger_path = CHROMA_DIR / "ingested_hashes.txt"
    # return empty set if ledger file does not exist
    if not ledger_path.exists():
        return set()
    return set(ledger_path.read_text().splitlines())

# writes a hash to the ledger file
def save_ingested_hash(hash: str) -> None:
    ledger_path = CHROMA_DIR / "ingested_hashes.txt"
    with open(ledger_path, "a") as f:
        f.write(f"{hash}\n")

if __name__ == "__main__":

    print("-" * 50)
    print("AVIONICS RAG INGEST SCRIPT")
    print("-" * 50)

    CHROMA_DIR.mkdir(exist_ok=True)
    ingested = load_ingested_hashes()

    # 1. Load and split documents
    # ----------------------------------------------------------------------------------------------------------
    print("[1/4] Loading and splitting documents...")
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    new_docs = []
    new_hashes = []

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

        # save all new pages and new file hashes
        new_docs.extend(pages)
        new_hashes.append(file_hash)

    print(f" - New docs to add: {len(new_docs)}")
    print(f" - New file hashes to add: {len(new_hashes)}")

    # close out of program if we dont have anything new to ingest
    if not new_docs:
        print(" - No new documents to add. Exiting....")
        exit(0)

    # 2. Create chunks
    # ----------------------------------------------------------------------------------------------------------
    print("[2/4] Creating chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(new_docs)
    print(f" - Total chunks created: {len(chunks)}")

    # 3. Create embeddings and persist to ChromaDB
    # ----------------------------------------------------------------------------------------------------------
    print("[3/4] Creating embeddings and persisting to ChromaDB...")
    print(f" - Model: {EMBED_MODEL}")
    print(f" - Collection name: {COLLECTION_NAME}")
    print(f" - ChromaDB directory: {CHROMA_DIR}")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    # if collection already exists, add to it. if collection is brand new, add_documents will create it
    vectorstore = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))
    vectorstore.add_documents(chunks)

    # 4. Record all hashes so we can skip these files in the next run
    # ----------------------------------------------------------------------------------------------------------        
    print("[4/4] Updating ingested hashes ledger...")
    for hash in new_hashes:
        save_ingested_hash(hash)

    print(f"\nDone! Ingested {len(new_hashes)} new docs, split into {len(chunks)} chunks.")