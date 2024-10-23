import psycopg2
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from app.core.config import get_settings


settings = get_settings()
MODEL_EMBEDDINGS="sentence-transformers/sentence-t5-xxl"

class VectorStoreDB:
    def __init__(self):
        self.conn = self.get_database_connection()
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.SCW_SECRET_KEY,
            openai_api_base=settings.SCW_GENERATIVE_APIs_ENDPOINT,
            model=MODEL_EMBEDDINGS,
            tiktoken_enabled=False
        )
        self.vector_store = self.get_vector_store()


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

    def add_object_key(self, obj_key):
        """Inserts a new object key into the 'object_loaded' table."""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO object_loaded (object_key) VALUES (%s)",
                (obj_key,),
            )
            self.conn.commit()

    def add_embeddings_to_store(self, chunks, context_chunks, obj_key):
        """Adds embeddings to the vector store and inserts object keys into the database."""
        embeddings_list = [self.embeddings.embed_query(chunk) for chunk in context_chunks]
        self.vector_store.add_embeddings(chunks, embeddings_list)
        for chunk in chunks:
            self.add_object_key(obj_key)
