from app.core.config import get_settings
from app.services.vector_store import VectorStoreDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..services.object_store import ObjectStore
from ..services.model import MODELService
from ..services.embed_doc import create_context_prompt
from ..models.chat import Message


settings = get_settings()
BATCH_SIZE=10


def get_context_chunks(doc_key, doc, chunks):
    context_chunks = []

    doc_with_title = "Page title: " + doc_key + "\n" + doc

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


def get_context_doc(doc_key, doc, chunks):
    context_chunks = []

    doc_with_title = "Page title: " + doc_key + "\n" + doc

    completion = MODELService(
        [
            Message(
                role="user",
                content=create_context_prompt(document_content=doc_with_title),
            )
        ]
    ).generate_response(rag_enable=False)

    for chunk in chunks:
        context_chunks.append(completion + chunk)

    return context_chunks

def bucket_to_vectorDB():
    object_store = ObjectStore()
    doc_titles_added = []

    for page in object_store.get_page_iterator():
        for obj in page.get("Contents", []):
            doc_titles_added.append(doc_to_vectorDB(obj["key"]))

    return doc_titles_added


def doc_to_vectorDB(doc_key):
    vector_store_DB = VectorStoreDB()
    object_store = ObjectStore()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        add_start_index=True,
        length_function=len,
        is_separator_regex=False,
    )

    response = vector_store_DB.check_object_loaded(doc_key)

    if response is None:
        doc = object_store.get_document(doc_key)
        chunks = text_splitter.split_text(doc)

        url = doc.split("|")[0]
        name = doc.split("|")[1]

        context = get_context_doc(doc_key, doc, chunks)
        print ("Adding to vector store: ", doc_key, " with ", len(chunks), " chunks")
        for i in range(0, len(chunks), BATCH_SIZE):
            print ("Batches: ", i, " to ", i + BATCH_SIZE, " of ", len(chunks), " chunks")

            batch_chunks = chunks[i:i + BATCH_SIZE]  
            batch_context = context[i:i + BATCH_SIZE]

            try:
                vector_store_DB.add_embeddings_to_store(
                    batch_chunks, batch_context, doc_key, url, name
                )
                pass
            except Exception as e:
                print(f"An error occurred: {e}")

    return doc_key
