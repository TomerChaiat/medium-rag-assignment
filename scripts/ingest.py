import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHUNK_SIZE, OVERLAP_RATIO
from app.embeddings import embed_text
from app.pinecone_utils import create_index_if_needed, get_index

BATCH_SIZE = 50
REQUIRED_COLUMNS = {"title", "text", "url", "authors", "timestamp", "tags"}


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[str]:
    if not text or not str(text).strip():
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(str(text))

    if not tokens:
        return []

    overlap = int(chunk_size * overlap_ratio)
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("chunk_size and overlap_ratio produce non-positive step size.")

    chunks = []

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk = encoding.decode(chunk_tokens).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(tokens):
            break

    return chunks


def clean_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def text_for_embedding(title: str, authors: str, tags: str, chunk: str) -> str:
    return f"""
Title: {title}
Authors: {authors}
Tags: {tags}

Passage:
{chunk}
""".strip()


def validate_columns(dataframe: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")


def ingest(csv_path: str, limit: int | None = None) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Reading CSV: {path}")
    dataframe = pd.read_csv(path)
    validate_columns(dataframe)

    if limit is not None:
        dataframe = dataframe.head(limit)
        print(f"Ingesting first {limit} rows.")
    else:
        print(f"Ingesting all {len(dataframe)} rows.")

    create_index_if_needed()
    index = get_index()
    batch = []
    vector_count = 0
    skipped_rows = 0

    for article_id, row in dataframe.iterrows():
        title = clean_value(row.get("title"))
        text = clean_value(row.get("text"))
        url = clean_value(row.get("url"))
        authors = clean_value(row.get("authors"))
        timestamp = clean_value(row.get("timestamp"))
        tags = clean_value(row.get("tags"))

        if not text:
            skipped_rows += 1
            continue

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP_RATIO)

        for chunk_index, chunk in enumerate(chunks):
            embedding = embed_text(text_for_embedding(title, authors, tags, chunk))
            vector_id = f"article_{article_id}_chunk_{chunk_index}"
            metadata = {
                "article_id": str(article_id),
                "title": title,
                "url": url,
                "authors": authors,
                "timestamp": timestamp,
                "tags": tags,
                "chunk": chunk,
                "chunk_index": chunk_index,
            }

            batch.append(
                {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata,
                }
            )

            if len(batch) >= BATCH_SIZE:
                index.upsert(vectors=batch)
                vector_count += len(batch)
                print(f"Uploaded {vector_count} vectors...")
                batch = []

        if (article_id + 1) % 25 == 0:
            print(f"Processed {article_id + 1} rows...")

    if batch:
        index.upsert(vectors=batch)
        vector_count += len(batch)
        print(f"Uploaded {vector_count} vectors...")

    print(f"Done. Uploaded {vector_count} vectors. Skipped {skipped_rows} empty rows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Medium articles into Pinecone.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to ingest.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(args.csv, args.limit)
