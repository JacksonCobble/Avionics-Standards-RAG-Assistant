import chromadb
import random
import os
from dotenv import load_dotenv
from openai import OpenAI

def getsamples() -> None:
    # try to connect to chromaDB after ingestion
    client = chromadb.PersistentClient(path="chroma_db/")
    collection = client.get_collection("avionics_standards")

    # informatics about db
    print(client.heartbeat())
    print(client.list_collections())
    print(collection.count())

    # get a random sample of chunks
    all_ids = collection.get(include=[])["ids"]
    random_ids = random.sample(all_ids, 10)
    sample = collection.get(ids=random_ids, include=["documents", "metadatas"])

    # print out examples
    for doc, meta in zip(sample["documents"], sample["metadatas"]):
        print(f"SOURCE: {meta['source']}  |  page: {meta['page']}")
        print(doc[:300])
        print("-" * 50)

    return

def query_db() -> None:
    # prep environment
    load_dotenv()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path="chroma_db/")
    collection = chroma.get_collection("avionics_standards")

    # embed a question
    question = "What is the maximum stub length allowed in MIL-STD-1553?"

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    # check size of response- should be 1536
    vector = response.data[0].embedding
    print(f"Dimensions: {len(vector)}")

    # query to find the 5 most similar chunks to the prompt
    results = collection.query(
    query_embeddings=[vector],
    n_results=5,
    include=["documents", "metadatas", "distances"]
    )

    # all information to check to see how well we can collect results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # print informatics
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        print(f"--- Chunk {i} ---")
        print(f"Source: {meta['source']}  |  Page: {meta['page']}")
        print(f"Distance: {dist:.4f}")
        print(doc[:300])
        print()

    return

def build_prompt(question: str) -> str:
    # prep environment
    load_dotenv()

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path="chroma_db/")
    collection = chroma.get_collection("avionics_standards")

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    vector = response.data[0].embedding

    # query to find the 5 most similar chunks to the prompt
    results = collection.query(
    query_embeddings=[vector],
    n_results=5,
    include=["documents", "metadatas", "distances"]
    )

    # all information to check to see how well we can collect results
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context_blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        source = meta['source']
        page = meta['page']
        context_blocks.append(f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc}")
    
    context = "\n\n---\n\n".join(context_blocks)
    
    prompt = f"""You are a technical expert on avionics data bus standards.
    Answer the question below using ONLY the provided context.
    If the context does not contain enough information, say so — do not speculate.
    When you cite a fact, reference its source and page like this: [SOURCE: filename, p.N].

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:"""

    return prompt

def query_chatgpt_with_prompt(prompt: str) -> None:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    #create a wrap around prompt string on gpt 4-0 mini. temperature being at 0 lowers randomness in vocabulary
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
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

    # get answer and print it
    answer = completion.choices[0].message.content
    print("\n=== LLM ANSWER ===")
    print(answer)
    return


if __name__ == "__main__":
    getsamples()
    query_db()

    prompt = build_prompt(question= "What is the maximum stub length allowed in MIL-STD-1553?")
    print(prompt)
    print("-" * 50)
    query_chatgpt_with_prompt(prompt)

    badprompt = build_prompt(question= "What is the maximum range of an AIM-9 Sidewinder missile?")
    print(badprompt)
    print("-" * 50)
    query_chatgpt_with_prompt(badprompt)
    
    print("Done Running Tests")