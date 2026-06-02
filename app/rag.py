from app.config import CHAT_MODEL, TOP_K
from app.embeddings import embed_text, get_openai_client
from app.pinecone_utils import get_index
from app.prompts import SYSTEM_PROMPT

UNKNOWN_ANSWER = "I don’t know based on the provided Medium articles data."


def _metadata_value(metadata: dict, key: str) -> str:
    value = metadata.get(key, "")
    return "" if value is None else str(value)


def retrieve_context(question: str) -> list[dict]:
    question_embedding = embed_text(question)
    index = get_index()
    query_top_k = min(TOP_K * 3, 30)

    results = index.query(
        vector=question_embedding,
        top_k=query_top_k,
        include_metadata=True,
    )

    context = []
    seen_article_ids = set()

    for match in results.get("matches", []):
        metadata = match.get("metadata") or {}
        article_id = _metadata_value(metadata, "article_id")

        if article_id in seen_article_ids:
            continue

        seen_article_ids.add(article_id)
        context.append(
            {
                "article_id": article_id,
                "title": _metadata_value(metadata, "title"),
                "chunk": _metadata_value(metadata, "chunk"),
                "score": float(match.get("score", 0.0)),
                "authors": _metadata_value(metadata, "authors"),
                "url": _metadata_value(metadata, "url"),
                "tags": _metadata_value(metadata, "tags"),
            }
        )

        if len(context) >= TOP_K:
            break

    return context


def build_user_prompt(question: str, context: list[dict]) -> str:
    context_sections = []

    for index, item in enumerate(context, start=1):
        context_sections.append(
            f"""
Context {index}
Article ID: {item["article_id"]}
Title: {item["title"]}
Authors: {item["authors"]}
Tags: {item["tags"]}
Score: {item["score"]:.4f}
Passage:
{item["chunk"]}
""".strip()
        )

    context_text = "\n\n---\n\n".join(context_sections) if context_sections else "No context retrieved."

    return f"""
User question:
{question}

Retrieved context:
{context_text}

Answer the user question strictly based on the retrieved context above.
Follow the user's requested output format exactly.
If the user asks for "only" something, do not add explanations, justifications, or extra text.
If the user asks to return only titles, return only the titles and no explanation.
If the user asks for exactly N articles, return exactly N distinct article titles when enough relevant articles are available.
If the retrieved context is insufficient, answer exactly:
{UNKNOWN_ANSWER}
""".strip()


def call_chat_model(system_prompt: str, user_prompt: str) -> str:
    response = get_openai_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content or ""


def answer_question(question: str) -> dict:
    context = retrieve_context(question)
    user_prompt = build_user_prompt(question, context)
    response = call_chat_model(SYSTEM_PROMPT, user_prompt)

    return {
        "response": response,
        "context": [
            {
                "article_id": item["article_id"],
                "title": item["title"],
                "chunk": item["chunk"],
                "score": item["score"],
            }
            for item in context
        ],
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }
