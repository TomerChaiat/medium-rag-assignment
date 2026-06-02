import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medium-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"

EMBEDDING_DIMENSION = 1536

CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 7


def require_env_vars() -> None:
    missing = []
    for name, value in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "PINECONE_API_KEY": PINECONE_API_KEY,
    }.items():
        if not value:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
