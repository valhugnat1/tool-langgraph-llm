import os
from functools import lru_cache
import psycopg2
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from app.core.config import get_settings

settings = get_settings()

@lru_cache()
def get_database_connection():
    return psycopg2.connect(
        database=settings.SCW_DB_NAME,
        user=settings.SCW_DB_USER,
        password=settings.SCW_DB_PASSWORD,
        host=settings.SCW_DB_HOST,
        port=settings.SCW_DB_PORT,
    )

@lru_cache()
def get_embeddings():
    return OpenAIEmbeddings(
        openai_api_key=settings.SCW_SECRET_KEY,
        openai_api_base=settings.SCW_GENERATIVE_APIs_ENDPOINT,
        model="sentence-transformers/sentence-t5-xxl",
        tiktoken_enabled=False,
    )

@lru_cache()
def get_vector_store():
    conn = get_database_connection()
    embeddings = get_embeddings()
    
    connection_string = (
        f"postgresql+psycopg2://{conn.info.user}:{conn.info.password}"
        f"@{conn.info.host}:{conn.info.port}/{conn.info.dbname}"
    )
    
    return PGVector(
        connection_string=connection_string,
        embeddings=embeddings,
    )