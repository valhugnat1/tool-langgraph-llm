from app.core.config import get_settings
from app.services.vector_store import VectorStoreDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..services.object_store import ObjectStore
from ..services.model import MODELService
from ..services.embed_doc import create_context_prompt
from ..models.chat import Message


settings = get_settings()


def get_context_chunks(obj, doc, chunks):
    context_chunks = []

    doc_with_title = "Page title: " + obj["Key"] + "\n" + doc

    for chunk in chunks:
        completion = MODELService(
            [
                Message(
                    role="user",
                    content=create_context_prompt(
                        document_content=doc_with_title, chunk_text=chunk
                    ),
                )
            ]
        ).generate_response(rag_enable=False)
        context_chunks.append(completion + chunk)
    return context_chunks


def bucket_to_vectorDB():
    object_store = ObjectStore()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        add_start_index=True,
        length_function=len,
        is_separator_regex=False,
    )

    doc_titles_added = []

    vector_store_DB = VectorStoreDB()

    for page in object_store.get_page_iterator():
        for obj in page.get("Contents", []):
            response = vector_store_DB.check_object_loaded(obj["Key"])

            if response is None :
                doc = object_store.get_document(obj)
                chunks = text_splitter.split_text(doc)
                lines = doc.splitlines()  # Split the content into lines

                url = lines[0] if len(lines) > 0 else ""
                name = lines[2] if len(lines) > 2 else ""

                context_chunks = get_context_chunks(obj, doc, chunks)

                try:
                    vector_store_DB.add_embeddings_to_store(
                        chunks, context_chunks, obj["Key"], url, name
                    )
                    doc_titles_added.append(obj["Key"])

                except Exception as e:
                    print(f"An error occurred: {e}")

    return doc_titles_added
