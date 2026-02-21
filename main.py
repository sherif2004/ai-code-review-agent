import os
import logging
import requests
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from fastapi.responses import HTMLResponse

# =========================================================
# CONFIG
# =========================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def health():
    return {
        "status": "running",
        "test_mode": TEST_MODE
    }

# =========================================================
# OPENROUTER LLM CALL (WITH DEBUG)
# =========================================================

def analyze_code_with_llm(text: str):

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        return "LLM key missing."

    prompt = f"""
You are a senior software engineer.

Review this code and give improvement suggestions:

{text}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://railway.app",
        "X-Title": "AI Code Review Agent"
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    logger.info("Sending request to OpenRouter...")
    logger.info(f"Model: {OPENROUTER_MODEL}")
    logger.info(f"Prompt length: {len(prompt)}")

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=data
    )

    logger.info(f"OpenRouter status code: {response.status_code}")

    if response.status_code != 200:
        logger.error("OpenRouter error response:")
        logger.error(response.text)
        return f"LLM failed: {response.text}"

    result = response.json()

    try:
        output = result["choices"][0]["message"]["content"]
        logger.info("LLM response received successfully.")
        return output
    except Exception:
        logger.error("Unexpected OpenRouter response format")
        logger.error(result)
        return "LLM returned unexpected format."

# =========================================================
# TEST LLM ENDPOINT
# =========================================================

@app.get("/test-llm")
def test_llm():
    """
    Directly test OpenRouter without GitHub.
    """

    sample_text = "def add(a,b): return a+b"

    result = analyze_code_with_llm(sample_text)

    return {
        "llm_response": result
    }

# =========================================================
# GITHUB HELPERS
# =========================================================

def fetch_pr_files(repo_full_name: str, pr_number: int):

    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set")
        return None

    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        logger.error("GitHub API error:")
        logger.error(response.text)
        return None

    return response.json()

def post_pr_comment(repo_full_name: str, pr_number: int, comment: str):

    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {"body": comment}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 201:
        logger.error("Failed to post comment:")
        logger.error(response.text)

# =========================================================
# BACKGROUND PROCESS
# =========================================================

def process_pr(payload: dict):

    action = payload.get("action")

    if action not in ["opened", "synchronize", "reopened"]:
        logger.info(f"Ignored action: {action}")
        return

    repo = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]

    logger.info(f"Processing PR #{pr_number} in {repo}")

    files = fetch_pr_files(repo, pr_number)

    if not files:
        return

    combined_patch = ""

    for file in files:
        patch = file.get("patch")
        if patch:
            combined_patch += f"\n\nFile: {file['filename']}\n{patch}"

    if not combined_patch:
        logger.info("No patch content found.")
        return

    if TEST_MODE:
        logger.info("TEST_MODE enabled — skipping LLM and comment posting.")
        return

    review = analyze_code_with_llm(combined_patch)

    post_pr_comment(repo, pr_number, review)

    logger.info("AI review posted successfully.")

# =========================================================
# WEBHOOK
# =========================================================
from fastapi import HTTPException

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):

    body = await request.body()

    # 1️⃣ Ensure body exists
    if not body:
    logger.info("Empty webhook request ignored.")
    return {"status": "ignored"}

    # 2️⃣ Ensure valid JSON
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format"
        )

    logger.info("Webhook received")

    # 3️⃣ Ensure it's a PR event
    if "pull_request" not in payload:
        return {"status": "not a pull request event"}

    background_tasks.add_task(process_pr, payload)

    return {"status": "processing"}

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



