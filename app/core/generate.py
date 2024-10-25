import time
import json
from app.services.model import MODELService
from app.services.embed_doc import source_clean_string

MODEL_RAG = "llama-3.1-8b-rag"
MODEL_VANILLA = "llama-3.1-8b-instruct"


def query_model(request):
    llm = MODELService(request.messages)

    if request.model == MODEL_RAG:
        data_response = llm.generate_response(True)
        sources = source_clean_string(data_response)
        return data_response["answer"] + sources
    elif request.model == MODEL_VANILLA:
        return llm.generate_response(False)
    else:
        raise Exception("Model not available")


async def stream_query_model(request: str):
    llm = MODELService(request.messages)
    i = 0

    if request.model == MODEL_RAG:
        stream = await llm.stream_response(True)
        chunk_sources = {}

        for response_chunk in stream:
            if "answer" in response_chunk.keys():
                chunk = {
                    "id": i,
                    "object": "chat.completion.chunk",
                    "created": time.time(),
                    "model": request.model,
                    "choices": [{"delta": {"content": response_chunk["answer"]}}],
                }
                i += 1
                yield f"data: {json.dumps(chunk)}\n\n"

            if "context" in response_chunk.keys():
                sources = source_clean_string(response_chunk)
                chunk_sources = {
                    "id": i,
                    "object": "chat.completion.chunk",
                    "created": time.time(),
                    "model": request.model,
                    "choices": [{"delta": {"content": sources}}],
                }

            i += 1

        if chunk_sources != {}: 
            yield f"data: {json.dumps(chunk_sources)}\n\n"
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
