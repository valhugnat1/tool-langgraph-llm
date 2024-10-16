from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import time
import asyncio

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, HTTPException, Request
app = FastAPI()


# Middleware to log requests with incorrect paths
class LogIncorrectPathsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 404:
            # Log the request path and other details
            print(f"Incorrect endpoint path: {request.url.path}, Method: {request.method}")
            # You can add more logging details if needed
        return response



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, replace "*" with specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


from fastapi.responses import Response

# Middleware to set Referrer-Policy to strict-origin-when-cross-origin
@app.middleware("http")
async def set_referrer_policy(request: Request, call_next):
    response = await call_next(request)
    
    # Add the Referrer-Policy header
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

app.add_middleware(LogIncorrectPathsMiddleware)

# An async generator that yields tokens as they are generated.
async def token_stream():
    tokens = ["token1", "token2", "token3", "token4"]  # Simulating token generation
    for token in tokens:
        await asyncio.sleep(1)  # Simulate delay
        yield token
    yield "end"  # Indicate the end of the stream

@app.post("/chat/completions")
async def stream_tokens():
    async def event_generator():
        async for token in token_stream():
            yield {"event": "message", "data": token}
    
    return EventSourceResponse(event_generator())


@app.get("/models")
async def models_list():

    return {
        "object": "list",
        "data": [
            {
            "id": "llama-3.1-8b-rag",
            "object": "model",
            "created": 1686935008,
            "owned_by": "organization-owner"
            },
        ],
        "object": "list"
        }

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}