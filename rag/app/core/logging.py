import logging
from typing import Optional
from lunary import LunaryCallbackHandler
from app.core.config import get_settings

settings = get_settings()

def setup_logging(level: Optional[str] = "INFO"):
    # Configure basic logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup Lunary handler
    lunary_handler = LunaryCallbackHandler(
        app_id=settings.LUNARY_PUBLIC_KEY
    )
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Add Lunary handler
    logger.addHandler(lunary_handler)
    
    return lunary_handler

# Create a logger instance for this module
logger = logging.getLogger(__name__)