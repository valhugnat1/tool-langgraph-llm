import time
import json
from app.services.model import MODELService
from app.services.embed_doc import source_clean_string
from ..services.vector_store import VectorStoreDB

def get_model_info (model): 

    model_list = VectorStoreDB().get_models_table()

    for model_info in model_list:
        if model_info["model_name"] == model:
            return model_info["model_type"] == "rag", model_info["base_model"]

    raise Exception("Model not available")


# No stream generation
def query_model(request):

    is_rag, base_model = get_model_info(request.model)
    llm = MODELService(request.messages, request.temperature, request.max_tokens, base_model)

    if (
        not is_rag
        or "Create a concise, 3-5 word title with an emoji as a title for the prompt in the given language."  # Specific case for title generation
        in str(request.messages[-1].content)
    ):
        return llm.generate_response(False)
    else:
        data_response = llm.generate_response(True)
        sources = source_clean_string(data_response)
        return data_response["answer"] + sources

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

    is_rag, base_model = get_model_info(request.model)
    llm = MODELService(request.messages, request.temperature, request.max_tokens, base_model)
    i = 0

    if not is_rag:
        stream = await llm.stream_response(False)

        for response_chunk in stream:
            chunk = get_chunk_struct(response_chunk.content, request.model, i)
            i += 1
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    else :
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