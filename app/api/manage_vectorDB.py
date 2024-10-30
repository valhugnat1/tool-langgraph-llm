from fastapi import APIRouter
from ..core.add_vector import bucket_to_vectorDB, doc_to_vectorDB
from ..models.file import DocRequest
router = APIRouter()


@router.get("/data_loader")
def data_loader():


    doc_titles_added = bucket_to_vectorDB()

    return {
        "status": "done", 
        "number_docs": len (doc_titles_added),
        "doc_titles_added": doc_titles_added
    }

@router.post("/doc_to_vectordb")
def doc_to_vectordb(request: DocRequest):

    res = []
    for key in request.keys:

        res.append(doc_to_vectorDB(doc_key=key))

    return {
        "status": "done",
        "result": res
    }