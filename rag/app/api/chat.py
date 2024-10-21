import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ..models.chat import ChatCompletionRequest, Message
from ..core.rag import _resp_async_generator, query_rag

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages:
        _, response_text = query_rag(str(request.messages[-1]))

        if request.stream:
            return StreamingResponse(
                _resp_async_generator(request), media_type="text/event-stream"
            )

        else:
            _, response_text = query_rag(str(request.messages[-1]))

            return {
                "id": "1337",
                "object": "chat.completion",
                "created": time.time(),
                "model": request.model,
                "choices": [
                    {"message": Message(role="assistant", content=str(response_text))}
                ],
            }



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
        ],
    }
