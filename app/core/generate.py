import time
import json
from app.services.model import MODELService
from app.services.embed_doc import source_clean_string

MODEL_RAG = "llama-3.1-8b-rag"
MODEL_VANILLA = "llama-3.1-8b-instruct"


# No stream generation
def query_model(request):
    llm = MODELService(request.messages, request.temperature, request.max_tokens)

    if (
        request.model == MODEL_VANILLA
        or "Create a concise, 3-5 word title with an emoji as a title for the prompt in the given language."  # Specific case for title generation
        in str(request.messages[-1].content)
    ):
        return llm.generate_response(False)
    elif request.model == MODEL_RAG:
        data_response = llm.generate_response(True)
        sources = source_clean_string(data_response)
        return data_response["answer"] + sources
    else:
        raise Exception("Model not available")

def get_chunk_struct(content, model, i):
    return {
        "id": i,
        "object": "chat.completion.chunk",
        "created": time.time(),
        "model": model,
        "choices": [{"delta": {"content": content}}],
    }

# Stream generation
async def stream_query_model(request: str):
    llm = MODELService(request.messages, request.temperature, request.max_tokens)
    i = 0

    if request.model == MODEL_VANILLA:
        stream = await llm.stream_response(False)

        for response_chunk in stream:
            chunk = get_chunk_struct(response_chunk.content, request.model, i)
            i += 1
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    elif request.model == MODEL_RAG:
        stream = await llm.stream_response(True)
        chunk_sources = {}

        for response_chunk in stream:
            if "answer" in response_chunk.keys():
                chunk = get_chunk_struct(response_chunk["answer"], request.model, i)

                i += 1
                yield f"data: {json.dumps(chunk)}\n\n"

            if "context" in response_chunk.keys():
                sources = source_clean_string(response_chunk)
                chunk_sources = get_chunk_struct(sources, request.model, i)

            i += 1

        if chunk_sources != {}:
            yield f"data: {json.dumps(chunk_sources)}\n\n"
        yield "data: [DONE]\n\n"

    else:
        raise Exception("Model not available")
