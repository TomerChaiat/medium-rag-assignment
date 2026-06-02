from functools import lru_cache

from openai import OpenAI

from app.config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, require_env_vars


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    require_env_vars()
    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def embed_text(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    response = get_openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding
