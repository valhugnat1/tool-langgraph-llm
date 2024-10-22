### Import and setup ###

import json
import os  # Importing os module for operating system functionalities
import time
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import logging
from lunary import LunaryCallbackHandler

import psycopg2


### Setup monitoring & logs ###

load_dotenv()  # This loads the environment variables from the .env file

handler = LunaryCallbackHandler(app_id=os.getenv("LUNARY_PUBLIC_KEY"))

logging.basicConfig(level=logging.INFO)

### Load Retriver ###

CHROMA_PATH = "chroma"  # old vector DB

conn = psycopg2.connect(
    database=os.getenv("SCW_DB_NAME"),
    user=os.getenv("SCW_DB_USER"),
    password=os.getenv("SCW_DB_PASSWORD"),
    host=os.getenv("SCW_DB_HOST"),
    port=os.getenv("SCW_DB_PORT"),
)

cur = conn.cursor()

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("SCW_SECRET_KEY"),
    openai_api_base=os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
    model="sentence-transformers/sentence-t5-xxl",
    tiktoken_enabled=False,
)


connection_string = f"postgresql+psycopg2://{conn.info.user}:{conn.info.password}@{conn.info.host}:{conn.info.port}/{conn.info.dbname}"
vector_store = PGVector(connection=connection_string, embeddings=embeddings)


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Function ###


def context_rag_fct(query_text):
    embedding_function = OpenAIEmbeddings()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")

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
        callbacks=[handler],
    )

    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text


async def _resp_async_generator(request: str):
    global vector_store

    llm = ChatOpenAI(
        base_url=os.getenv("SCW_GENERATIVE_APIs_ENDPOINT"),
        api_key=os.getenv("SCW_SECRET_KEY"),
        model="llama-3.1-8b-instruct",
    )

    prompt = hub.pull("rlm/rag-prompt")
    retriever = vector_store.as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    i = 0
    for response_chunk in rag_chain.stream(str(request.messages[-1])):

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
