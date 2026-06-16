from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2) # the users question in string format
    n_results: int = 5 # how many chunks to pull from ChromaDB for context- default 5

class Source(BaseModel):
    doc: str = "" # source doc filename
    page: int = 0 # page num in source doc

class QueryResponse(BaseModel):
    question: str = Field(..., min_length=2) # the users question in string format
    answer: str = Field(..., min_length=2)   # the generated answer from the model
    sources: list[Source] = Field(default_factory=list) # list of sources used to generate the answer
    chunks_used: int = 0 # how many chunks were fed to the LLM
    latency_ms: float = 0.0 # how long the request took in milliseconds