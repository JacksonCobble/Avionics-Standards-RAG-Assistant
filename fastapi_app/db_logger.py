import json
import logging
from typing import Optional
from models import QueryResponse

import psycopg
from psycopg_pool import ConnectionPool

from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

# type hint for pool
_pool: Optional[ConnectionPool] = None

# "singleton" for connection pool
def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
                raise RuntimeError("DATABSE URL not in .env")
        _pool = ConnectionPool(conninfo=database_url, min_size=1,max_size=10, open=True)
        logger.info("Postgres connection pool created")
    return _pool


# INIT DB
_CREATE_TABLE =  """
CREATE TABLE IF NOT EXISTS query_logs (
    id  SERIAL PRIMARY KEY,
    question    TEXT    NOT NULL,
    answer      TEXT    NOT NULL,
    sources     JSONB,
    chunks_used INTEGER,
    latency_ms  FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
    )"""

def init_db() -> None:
    with _get_pool().connection() as conn:
        conn.execute(_CREATE_TABLE)
    logger.info("query_logs table ready")


# log a query given a queryresponse
def log_query(response: QueryResponse) -> None:
    # sql to insert a full row into table
    # IMPORTANT- :: is the way postgres does typecasting
    _insert = """
    INSERT INTO query_logs (question, answer, sources, chunks_used, latency_ms)
    VALUES (%s, %s, %s::jsonb, %s, %s)"""

    # call insert function passing in response information
    try:
        with _get_pool().connection() as conn:
            conn.execute(_insert, (response.question,
                        response.answer,
                        # response.sources is a list[source] objects defined in model. 
                        # dump all source objects into JSON, then turn list into JSON string
                        json.dumps([s.model_dump() for s in response.sources]),
                        response.chunks_used,
                        response.latency_ms))
    
    # throw errors to logger
    except Exception as e:
         logger.error("Failed to log query: %s", e, exc_info=True)

    return