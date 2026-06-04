from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from app.rag import answer_question

app = FastAPI(title="Medium Article RAG Assistant")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Medium RAG Assistant</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f6f7fb;
      margin: 0;
      padding: 0;
      color: #222;
    }

    .container {
      max-width: 900px;
      margin: 60px auto;
      padding: 32px;
      background: white;
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }

    h1 {
      margin-top: 0;
      font-size: 34px;
      color: #111;
    }

    h3 {
      margin-bottom: 8px;
      color: #111;
    }

    p {
      color: #555;
      line-height: 1.6;
    }

    .stats {
      margin: 20px 0 24px 0;
      padding: 14px 16px;
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      font-size: 14px;
      color: #333;
    }

    textarea {
      width: 100%;
      min-height: 110px;
      padding: 14px;
      font-size: 16px;
      border: 1px solid #ddd;
      border-radius: 12px;
      resize: vertical;
      box-sizing: border-box;
    }

    button {
      margin-top: 14px;
      padding: 12px 22px;
      font-size: 16px;
      border: none;
      border-radius: 10px;
      background: #111;
      color: white;
      cursor: pointer;
    }

    button:hover {
      background: #333;
    }

    button:disabled {
      background: #888;
      cursor: not-allowed;
    }

    .answer {
      margin-top: 28px;
      padding: 20px;
      background: #f2f4f8;
      border-radius: 12px;
      white-space: pre-wrap;
      line-height: 1.6;
    }

    .context {
      margin-top: 24px;
    }

    .context-item {
      padding: 14px;
      border: 1px solid #eee;
      border-radius: 10px;
      margin-top: 10px;
      background: #fff;
    }

    .score {
      color: #777;
      font-size: 13px;
    }

    .augmented-prompt {
      margin-top: 24px;
    }

    .prompt-title {
      margin-top: 20px;
      margin-bottom: 8px;
      font-weight: bold;
      color: #111;
    }

    .prompt-box {
      margin-top: 8px;
      padding: 16px;
      background: #111827;
      color: #e5e7eb;
      border-radius: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
      font-family: monospace;
      font-size: 13px;
      line-height: 1.5;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Medium RAG Assistant</h1>
    <p>
      Ask questions about the Medium articles dataset. The assistant answers only
      from retrieved Medium article context, without relying on external knowledge.
    </p>

    <div id="stats" class="stats">
      Loading RAG configuration...
    </div>

    <textarea id="question" placeholder="Type your question here..."></textarea>
    <br />
    <button id="askButton" onclick="askQuestion()">Ask</button>

    <div id="answer" class="answer" style="display:none;"></div>
    <div id="context" class="context"></div>
    <div id="augmentedPrompt" class="augmented-prompt"></div>
  </div>

  <script>
    async function loadStats() {
      const statsDiv = document.getElementById("stats");

      try {
        const response = await fetch("/api/stats");

        if (!response.ok) {
          throw new Error("Failed to load stats");
        }

        const stats = await response.json();

        statsDiv.innerHTML = `
          <strong>RAG configuration:</strong>
          Chunk size: ${stats.chunk_size} tokens |
          Overlap ratio: ${stats.overlap_ratio} |
          Top-k: ${stats.top_k}
        `;
      } catch (error) {
        statsDiv.innerText = "Could not load RAG configuration.";
      }
    }

    async function askQuestion() {
      const question = document.getElementById("question").value.trim();
      const button = document.getElementById("askButton");
      const answerDiv = document.getElementById("answer");
      const contextDiv = document.getElementById("context");
      const augmentedPromptDiv = document.getElementById("augmentedPrompt");

      if (!question) {
        alert("Please enter a question.");
        return;
      }

      button.disabled = true;
      button.innerText = "Thinking...";
      answerDiv.style.display = "block";
      answerDiv.innerText = "Loading...";
      contextDiv.innerHTML = "";
      augmentedPromptDiv.innerHTML = "";

      try {
        const response = await fetch("/api/prompt", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ question })
        });

        if (!response.ok) {
          throw new Error("Request failed with status " + response.status);
        }

        const data = await response.json();

        answerDiv.innerText = data.response || "No response.";

        if (data.context && data.context.length > 0) {
          contextDiv.innerHTML = "<h3>Retrieved Context</h3>";

          data.context.forEach((item) => {
            const div = document.createElement("div");
            div.className = "context-item";
            div.innerHTML = `
              <strong>${escapeHtml(item.title || "Untitled")}</strong>
              <div class="score">Article ID: ${escapeHtml(item.article_id || "")} | Score: ${item.score}</div>
              <p>${escapeHtml((item.chunk || "").slice(0, 450))}...</p>
            `;
            contextDiv.appendChild(div);
          });
        }

        if (data.Augmented_prompt) {
          augmentedPromptDiv.innerHTML = "<h3>Augmented Prompt</h3>";

          const systemTitle = document.createElement("div");
          systemTitle.className = "prompt-title";
          systemTitle.innerText = "System:";

          const systemBox = document.createElement("div");
          systemBox.className = "prompt-box";
          systemBox.innerText = data.Augmented_prompt.System || "";

          const userTitle = document.createElement("div");
          userTitle.className = "prompt-title";
          userTitle.innerText = "User:";

          const userBox = document.createElement("div");
          userBox.className = "prompt-box";
          userBox.innerText = data.Augmented_prompt.User || "";

          augmentedPromptDiv.appendChild(systemTitle);
          augmentedPromptDiv.appendChild(systemBox);
          augmentedPromptDiv.appendChild(userTitle);
          augmentedPromptDiv.appendChild(userBox);
        }
      } catch (error) {
        answerDiv.innerText = "Error: " + error.message;
      } finally {
        button.disabled = false;
        button.innerText = "Ask";
      }
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    loadStats();
  </script>
</body>
</html>
"""


@app.get("/api/prompt", response_class=HTMLResponse)
def prompt_docs():
    return home()



class PromptRequest(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(request: PromptRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return answer_question(question)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }
