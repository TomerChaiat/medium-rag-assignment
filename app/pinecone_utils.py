from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from app.config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    require_env_vars,
)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    require_env_vars()
    return Pinecone(api_key=PINECONE_API_KEY)


def create_index_if_needed() -> None:
    pc = get_pinecone_client()
    indexes = pc.list_indexes()
    existing_indexes = indexes.names() if hasattr(indexes, "names") else [idx["name"] for idx in indexes]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )


def get_index():
    return get_pinecone_client().Index(PINECONE_INDEX_NAME)
