from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import chat, health, models
from .middleware.logging import LogIncorrectPathsMiddleware

def create_application() -> FastAPI:
    app = FastAPI(title="OpenAI-compatible RAG API")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(LogIncorrectPathsMiddleware)
    
    # Include routers
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)
    
    return app

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)