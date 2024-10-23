from fastapi import APIRouter
from ..core.add_vector import bucket_to_vectorDB
router = APIRouter()


@router.get("/data_loader")
def data_loader():


    doc_titles_added = bucket_to_vectorDB()

    return {
        "status": "done", 
        "number_docs": len (doc_titles_added),
        "doc_titles_added": doc_titles_added
    }
