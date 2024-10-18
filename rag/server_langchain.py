### Import and setup ###

import asyncio
import json
import os  # Importing os module for operating system functionalities
import shutil  # Importing shutil module for high-level file operations
import time
from typing import List, Optional

from dotenv import load_dotenv  # Importing dotenv to get API key from .env file
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI  # Import OpenAI LLM
from langchain.document_loaders.pdf import (
    PyPDFDirectoryLoader,
)  # Importing PDF loader from Langchain
from langchain.embeddings import (
    OpenAIEmbeddings,
)  # Importing OpenAI embeddings from Langchain
from langchain.schema import Document  # Importing Document schema from Langchain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Importing text splitter from Langchain
from langchain.vectorstores.chroma import (
    Chroma,
)  # Importing Chroma vector store from Langchain
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

# from langchain_openai import ChatOpenAI
# from lunary import LunaryCallbackHandler
# handler = LunaryCallbackHandler()


# Configure the logging system (you can customize this part to suit your needs)
logging.basicConfig(level=logging.INFO)

load_dotenv()  # This loads the environment variables from the .env file

# Now you can access the variables like this:
openai_api_key = os.getenv("OPENAI_API_KEY")


### data model ###


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


app = FastAPI(title="OpenAI-compatible API")


# Middleware to log requests with incorrect paths
class LogIncorrectPathsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 404:
            # Log the request path and other details
            print(
                f"Incorrect endpoint path: {request.url.path}, Method: {request.method}"
            )
            # You can add more logging details if needed
        return response


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins, replace "*" with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


### Function ###


def context_rag_fct(query_text):
    # embedding_function = OpenAIEmbeddings()
    embedding_function = OpenAIEmbeddings(
        openai_api_base="https://api.scaleway.ai/v1",
        openai_api_key=os.getenv("SCW_SECRET_KEY"),
    )

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    return prompt, results


def query_rag(query_text):
    prompt, results = context_rag_fct(query_text)

    model = ChatOpenAI(
        openai_api_base="https://api.scaleway.ai/v1",
        openai_api_key=os.getenv("SCW_SECRET_KEY"),
        model_name="llama-3.1-8b-instruct",
    )  # , callbacks=[handler]

    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text


async def _resp_async_generator(request: str):
    # Enable streaming in the model call
    model = ChatOpenAI(
        openai_api_base="https://api.scaleway.ai/v1",
        openai_api_key=os.getenv("SCW_SECRET_KEY"),
        model_name="llama-3.1-8b-instruct",
        streaming=True,
    )  # , callbacks=[handler]

    prompt, results = context_rag_fct(str(request.messages[-1]))

    i = 0
    for chunk in model.stream(prompt):
        response_chunk = chunk.content

        chunk = {
            "id": i,
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": "llama-3.1-8b-rag",
            "choices": [{"delta": {"content": response_chunk}}],
        }
        i += 1
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


### Server fastapi ###
#### Middleware ####

app.add_middleware(LogIncorrectPathsMiddleware)


# Middleware to set Referrer-Policy to strict-origin-when-cross-origin
@app.middleware("http")
async def set_referrer_policy(request: Request, call_next):
    response = await call_next(request)

    # Add the Referrer-Policy header
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    return response


# Middleware to log the query parameters of each request
@app.middleware("http")
async def log_query_params(request: Request, call_next):
    # Get the full URL and query parameters
    query_params = dict(request.query_params)

    # Log the query parameters
    logging.info(f"Request URL: {request.url} - Query params: {query_params}")

    # Proceed with the request
    response = await call_next(request)
    return response


#### Route ####


@app.post("/chat/completions")
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


@app.get("/models")
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
        "object": "list",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


### Main ###


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
