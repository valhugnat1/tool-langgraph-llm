from setuptools import setup, find_packages

setup(
    name="rag-api",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "langchain",
        "langchain-community",
        "langchain-openai",
        "python-dotenv",
        # Add other dependencies
    ],
)