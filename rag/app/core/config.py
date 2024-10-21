from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    SCW_DB_NAME: str
    SCW_DB_USER: str
    SCW_DB_PASSWORD: str
    SCW_DB_HOST: str
    SCW_DB_PORT: str
    SCW_SECRET_KEY: str
    SCW_GENERATIVE_APIs_ENDPOINT: str
    LUNARY_PUBLIC_KEY: str
    OPENAI_API_KEY: str
    SCW_ACCESS_KEY: str
    SCW_BUCKET_NAME: str
    SCW_REGION: str
    SCW_BUCKET_ENDPOINT: str
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()