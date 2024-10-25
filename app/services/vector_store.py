import psycopg2
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from app.core.config import get_settings
import json
from datetime import datetime
from langchain_core.runnables import chain

settings = get_settings()
MODEL_EMBEDDINGS = "sentence-transformers/sentence-t5-xxl"


class VectorStoreDB:
    def __init__(self):
        self.conn = self.get_database_connection()
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.SCW_SECRET_KEY,
            openai_api_base=settings.SCW_GENERATIVE_APIs_ENDPOINT,
            model=MODEL_EMBEDDINGS,
            tiktoken_enabled=False,
        )
        self.vector_store = self.get_vector_store()

    def get_retriever(self):
        @chain
        def retriever(query):
            docs, scores = zip(*self.vector_store.similarity_search_with_score(query))
            filtered_docs = [doc for doc, score in zip(docs, scores) if score <= 0.2]

            # Only store scores for documents that pass the threshold
            for doc in filtered_docs:
                doc.metadata["score"] = doc.metadata.get("score", 0.0)

            return filtered_docs

        return retriever

    def get_database_connection(self):
        """Establishes a database connection."""
        return psycopg2.connect(
            database=settings.PGDATABASE,
            user=settings.PGUSER,
            password=settings.PGPASSWORD,
            host=settings.PGHOST,
            port=settings.PGPORT,
        )

    def get_vector_store(self):
        """Creates a PGVector object with connection string and embeddings."""
        connection_string = (
            f"postgresql+psycopg2://{self.conn.info.user}:{self.conn.info.password}"
            f"@{self.conn.info.host}:{self.conn.info.port}/{self.conn.info.dbname}"
        )
        return PGVector(
            connection=connection_string,
            embeddings=self.embeddings,
        )

    def check_object_loaded(self, obj_key):
        """Checks if the object with the specified key exists in the 'object_loaded' table."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT object_key FROM object_loaded WHERE object_key = %s",
                (obj_key,),
            )
            response = cur.fetchone()
        return response

    def add_object_key(self, obj_key, metadata_list):
        """Inserts a new object key into the 'object_loaded' table."""

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO object_loaded (object_key, metadata) VALUES (%s, %s)",
                    (obj_key, json.dumps(metadata_list)),
                )
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()  # Rollback in case of error
            raise e

    def add_embeddings_to_store(self, chunks, context_chunks, obj_key, url, name):
        """Adds embeddings to the vector store and inserts object keys into the database."""
        try:
            embeddings_list = [
                self.embeddings.embed_query(chunk) for chunk in context_chunks
            ]
            metadata_list = self.get_metadata_file(obj_key, context_chunks, url, name)
            self.vector_store.add_embeddings(
                chunks, embeddings_list, metadatas=metadata_list
            )

            self.add_object_key(obj_key, metadata_list)

        except Exception as e:
            print(f"Error adding embeddings or object key: {e}")
            raise e

    def get_metadata_file(self, obj_key, context_chunks, url, name):
        return [
            {
                "chunk_id": idx,
                "source": obj_key,
                "timestamp": datetime.now().isoformat(),
                "chunk_size": len(chunk),
                "url": url,
                "name": name,
                "position": idx * len(chunk),
            }
            for idx, chunk in enumerate(context_chunks)
        ]
