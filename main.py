import os
import requests
from fastapi import FastAPI, Request, BackgroundTasks
from openai import OpenAI

app = FastAPI()

# Environment variables (set in Railway dashboard)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not set")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------
# GitHub API Functions
# -------------------------

def fetch_pr_diff(repo_full_name: str, pr_number: int):
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.diff"
    }

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        print("Failed to fetch diff:", response.text)
        return None

    return response.text


def post_pr_comment(repo_full_name: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    data = {"body": body}

    response = requests.post(url, headers=headers, json=data, timeout=10)

    print("Comment response:", response.status_code)


# -------------------------
# LLM Review
# -------------------------

def review_with_llm(diff_text: str):
    prompt = f"""
You are a senior software engineer.

Review the following GitHub Pull Request diff.

Provide:
1. Code issues (if any)
2. Improvements
3. Refactoring suggestions
4. A short summary

Return clean markdown.

Diff:
{diff_text[:6000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------
# Background Processing
# -------------------------

def process_pull_request(payload: dict):
    action = payload.get("action")

    # Only react to relevant PR events
    if action not in ["opened", "synchronize", "reopened"]:
        print("Ignored action:", action)
        return

    repo = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]

    print(f"Processing PR #{pr_number} in {repo}")

    # 1️⃣ Fetch diff
    diff = fetch_pr_diff(repo, pr_number)
    if not diff:
        print("No diff found")
        return

    print("Diff fetched successfully")

    # 2️⃣ Send to LLM
    review = review_with_llm(diff)

    print("LLM review generated")

    # 3️⃣ Post comment
    post_pr_comment(repo, pr_number, review)

    print("Comment posted successfully")


# -------------------------
# FastAPI Routes
# -------------------------

@app.get("/")
def health_check():
    return {"status": "AI Code Review Agent Running"}


@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()

    # Run heavy logic in background to avoid GitHub timeout
    background_tasks.add_task(process_pull_request, payload)

    return {"status": "processing"}
