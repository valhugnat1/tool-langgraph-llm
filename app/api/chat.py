import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ..models.chat import ChatCompletionRequest, Message
from ..core.generate import stream_query_model, query_model

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.messages:
        if request.stream:
            return StreamingResponse(
                stream_query_model(request), media_type="text/event-stream"
            )

        else:
            response_text = query_model(request)

            return {
                "id": "1337",
                "object": "chat.completion",
                "created": time.time(),
                "model": request.model,
                "choices": [
                    {"message": Message(role="assistant", content=str(response_text))}
                ],
            }