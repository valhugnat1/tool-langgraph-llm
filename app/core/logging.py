from lunary import LunaryCallbackHandler
from app.core.config import get_settings

settings = get_settings()


def setup_lunary():
    # Setup Lunary handler
    lunary_handler = LunaryCallbackHandler(app_id=settings.LUNARY_PUBLIC_KEY)

    return lunary_handler
