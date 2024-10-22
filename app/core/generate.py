import time
import json
from app.services.vector_store import get_vector_store
from app.services.model import MODELService

MODEL_RAG = "llama-3.1-8b-rag"
MODEL_VANILLA = "llama-3.1-8b-instruct"


def query_rag(request):
    llm = MODELService(get_vector_store(), request)

    if request.model == MODEL_RAG:
        return llm.generate_response(True)
    elif request.model == MODEL_VANILLA:
        return llm.generate_response(False)
    else:
        raise Exception("Model not available")


async def stream_query_rag(request: str):
    global vector_store

    llm = MODELService(get_vector_store(), request)
    i = 0

    if request.model == MODEL_RAG:
        stream = await llm.stream_response(True)

        for response_chunk in stream:
            chunk = {
                "id": i,
                "object": "chat.completion.chunk",
                "created": time.time(),
                "model": request.model,
                "choices": [{"delta": {"content": response_chunk}}],
            }
            i += 1
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    elif request.model == MODEL_VANILLA:
        stream = await llm.stream_response(False)

        for response_chunk in stream:
            chunk = {
                "id": i,
                "object": "chat.completion.chunk",
                "created": time.time(),
                "model": request.model,
                "choices": [{"delta": {"content": response_chunk.content}}],
            }
            i += 1
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    else:
        raise Exception("Model not available")
