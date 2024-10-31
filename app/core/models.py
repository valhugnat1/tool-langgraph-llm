from app.services.vector_store import VectorStoreDB

def query_model():
    vectorDB = VectorStoreDB()

    return vectorDB.get_models_table()
