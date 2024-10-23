from fastapi import APIRouter, Response, status
from app.services.vector_store import VectorStoreDB

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@router.get("/health/db")
async def database_health_check():
    """Check database connection"""
    try:
        conn = VectorStoreDB().conn
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return Response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": str(e)}
        )

@router.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "RAG API Service",
        "version": "1.0.0",
        "docs": "None"
    }