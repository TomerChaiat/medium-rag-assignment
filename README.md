# Medium Article RAG Assistant

Python implementation of the Individual Assignment: Medium Article RAG Assistant.

The app answers questions strictly from a Medium articles CSV dataset by retrieving article chunks from Pinecone, building an augmented prompt, and calling the required OpenAI-compatible course chat model.

## Project Structure

```text
medium-rag-assignment/
├── api/index.py
├── app/
├── scripts/ingest.py
├── data/.gitkeep
├── tests/test_basic.py
├── requirements.txt
├── vercel.json
├── .env.example
├── .gitignore
└── README.md
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template and fill in your real values:

```bash
cp .env.example .env
```

Put the dataset at:

```text
data/medium-english-50mb.csv
```

Do not commit the CSV file. It is ignored by `.gitignore`.

## Environment Variables

```env
OPENAI_API_KEY=your_4UHRUIN_api_key_here
OPENAI_BASE_URL=your_4UHRUIN_base_url_here

PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=medium-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Ingestion

Start with a small subset to control cost:

```bash
python scripts/ingest.py --csv data/medium-english-50mb.csv --limit 100
```

After verifying the system, ingest the full dataset:

```bash
python scripts/ingest.py --csv data/medium-english-50mb.csv
```

The ingestion script chunks each article by tokens, embeds each chunk with `4UHRUIN-text-embedding-3-small`, and uploads vectors with article metadata to Pinecone. Vector IDs are deterministic, using `article_{article_id}_chunk_{chunk_index}`.

## Run Locally

```bash
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

## Example API Calls

Stats:

```bash
curl http://127.0.0.1:8000/api/stats
```

Expected response:

```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 7
}
```

Prompt:

```bash
curl -X POST http://127.0.0.1:8000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "List exactly 3 articles about education. Return only the titles."}'
```

Response shape:

```json
{
  "response": "Natural language answer from the model.",
  "context": [
    {
      "article_id": "123",
      "title": "Sample article title",
      "chunk": "Retrieved article chunk...",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```

## Chosen RAG Hyperparameters

Chosen hyperparameters:
- chunk_size = 512 tokens
- overlap_ratio = 0.2
- top_k = 7

Rationale:
A chunk size of 512 tokens keeps retrieved passages focused while preserving enough context from each Medium article. An overlap ratio of 0.2 reduces the risk of splitting important ideas across chunk boundaries. top_k=7 provides enough evidence for the language model while avoiding unnecessary context expansion and cost.

The constraints are enforced in `app/config.py`:

- `chunk_size <= 1024`
- `overlap_ratio <= 0.3`
- `top_k <= 30`

## Deployment Notes

The first deployment target is Vercel. The `api/index.py` file exposes the FastAPI app and `vercel.json` rewrites `/api/*` traffic to the serverless entrypoint.

If Vercel Python deployment causes issues, the same FastAPI app can be deployed to Render or Railway with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Keep the Pinecone index active until grading is complete, otherwise `/api/prompt` will not be able to retrieve the article context.

## Tests

Run:

```bash
pytest
```

The included tests do not call external APIs.
