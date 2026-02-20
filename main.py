import os
import requests
from fastapi import FastAPI, Request, BackgroundTasks
from dotenv import load_dotenv

# Load environment variables from .env (for local dev)
load_dotenv()

app = FastAPI()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not set in environment variables")


def fetch_pr_diff(repo_full_name: str, pr_number: int):
    """
    Fetch the PR diff from GitHub API.
    """

    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.diff"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            print("GitHub API error:", response.status_code, response.text)
            return None

        return response.text

    except requests.RequestException as e:
        print("Request failed:", e)
        return None


def process_pr(payload: dict):
    """
    Heavy processing runs in background
    so webhook returns immediately.
    """

    try:
        action = payload.get("action")

        if action not in ["opened", "synchronize","reopen"]:
            print("Ignored action:", action)
            return

        repo = payload["repository"]["full_name"]
        pr_number = payload["pull_request"]["number"]

        print(f"Processing PR #{pr_number} in {repo}")

        diff = fetch_pr_diff(repo, pr_number)

        if diff:
            print("Diff extracted successfully")
            print("Diff length:", len(diff))
            print(diff[:800])  # Limit log size
        else:
            print("No diff returned")

    except Exception as e:
        print("Error in process_pr:", e)


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook endpoint.
    Must return fast to avoid GitHub timeout.
    """

    print("Webhook triggered")

    try:
        payload = await request.json()
    except Exception as e:
        print("Invalid JSON:", e)
        return {"error": "invalid json"}

    background_tasks.add_task(process_pr, payload)

    return {"status": "processing"}
