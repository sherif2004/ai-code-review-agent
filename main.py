import os
import logging
import requests
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn

# --------------------
# Config
# --------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --------------------
# Health
# --------------------
@app.get("/")
def health():
    return {"status": "running"}

# --------------------
# GitHub: Fetch PR Files
# --------------------
def fetch_pr_files(repo_full_name: str, pr_number: int):
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        logger.error("Failed to fetch PR files")
        return None

    return response.json()

# --------------------
# OpenRouter LLM Call
# --------------------
def analyze_code_with_llm(patch_text: str):

    prompt = f"""
You are a senior software engineer performing a professional code review.

Analyze this GitHub diff and provide:
- Potential bugs
- Code quality issues
- Performance improvements
- Security concerns
- Clear improvement suggestions

Diff:
{patch_text}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=data)

    if response.status_code != 200:
        logger.error("OpenRouter error: %s", response.text)
        return "LLM analysis failed."

    result = response.json()
    return result["choices"][0]["message"]["content"]

# --------------------
# GitHub: Post Comment
# --------------------
def post_pr_comment(repo_full_name: str, pr_number: int, comment: str):

    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {"body": comment}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 201:
        logger.error("Failed to post PR comment: %s", response.text)

# --------------------
# Background Processing
# --------------------
def process_pr(payload: dict):

    action = payload.get("action")

    if action not in ["opened", "synchronize", "reopened"]:
        logger.info("Ignored action: %s", action)
        return

    repo = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]

    logger.info("Processing PR #%s in %s", pr_number, repo)

    files = fetch_pr_files(repo, pr_number)

    if not files:
        return

    combined_patch = ""

    for file in files:
        patch = file.get("patch")
        if patch:
            combined_patch += f"\n\nFile: {file['filename']}\n{patch}"

    if not combined_patch:
        logger.info("No patch content to analyze.")
        return

    review = analyze_code_with_llm(combined_patch)

    post_pr_comment(repo, pr_number, review)

    logger.info("AI review posted successfully.")

# --------------------
# Webhook Endpoint
# --------------------
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):

    try:
        payload = await request.json()
    except Exception:
        return {"error": "invalid json"}

    logger.info("Webhook received")

    background_tasks.add_task(process_pr, payload)

    return {"status": "processing"}

# --------------------
# Railway Entry
# --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
