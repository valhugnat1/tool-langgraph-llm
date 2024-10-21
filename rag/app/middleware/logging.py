import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

class LogIncorrectPathsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        if response.status_code == 404:
            logger.warning(
                f"Incorrect endpoint accessed - Path: {request.url.path}, "
                f"Method: {request.method}, Client: {request.client.host}"
            )
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        logger.debug(f"Headers: {request.headers}")
        
        # Process request and get response
        response = await call_next(request)
        
        # Log response
        logger.info(f"Response: {response.status_code}")
        
        return response