from fastapi import APIRouter

router = APIRouter()


@router.get("/models")
async def models_list():
    return {
        "object": "list",
        "data": [
            {
                "id": "llama-3.1-8b-rag",
                "object": "model",
                "created": 1686935008,
                "owned_by": "organization-owner",
            },
            {
                "id": "llama-3.1-8b-instruct",
                "object": "model",
                "created": 1686935008,
                "owned_by": "organization-owner",
            },
        ],
    }
