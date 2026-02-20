import os
import requests
from fastapi import FastAPI, Request, BackgroundTasks
from openai import OpenAI

app = FastAPI()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def fetch_pr_diff(repo, pr_number):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.diff"
    }

    r = requests.get(url, headers=headers, timeout=5)
    if r.status_code != 200:
        print("GitHub error:", r.text)
        return None

    return r.text


def post_comment(repo, pr_number, body):
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    data = {"body": body}

    r = requests.post(url, headers=headers, json=data)
    print("Comment status:", r.status_code)


def review_with_llm(diff):
    prompt = f"""
    You are a senior software engineer.

    Review the following GitHub PR diff.

    Provide:
    1. Issues
    2. Improvements
    3. Suggested refactoring (if needed)

    Return structured markdown.

    Diff:
    {diff[:6000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


def process_pr(payload):
    action = payload.get("action")

    if action not in ["opened", "synchronize", "reopened"]:
        return

    repo = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]

    print("Processing PR:", pr_number)

    diff = fetch_pr_diff(repo, pr_number)
    if not diff:
        return

    review = review_with_llm(diff)

    post_comment(repo, pr_number, review)


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    background_tasks.add_task(process_pr, payload)
    return {"status": "processing"}
