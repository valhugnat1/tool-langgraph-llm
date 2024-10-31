from fastapi import APIRouter
from ..core.models import query_model

router = APIRouter()


@router.get("/models")
async def models_list():

    print (query_model())

    return {
        "object": "list",
        "data": [
            {
                "id": model_info[0],
                "object": "model",
                "created": 1686935008,
                "owned_by": "organization-owner",
            }
            for model_info in query_model()
        ],
    }
