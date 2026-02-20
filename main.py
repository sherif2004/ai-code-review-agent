import os
import hmac
import hashlib
import logging
import requests
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from dotenv import load_dotenv
import uvicorn

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not set in environment variables")

if not GITHUB_WEBHOOK_SECRET:
    raise ValueError("GITHUB_WEBHOOK_SECRET not set in environment variables")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()


# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def health():
    return {"status": "running"}


# ----------------------------
# Verify GitHub Signature
# ----------------------------
def verify_signature(payload_body: bytes, signature_header: str):
    if not signature_header:
        return False

    sha_name, signature = signature_header.split("=")
    if sha_name != "sha256":
        return False

    mac = hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        msg=payload_body,
        digestmod=hashlib.sha256,
    )

    return hmac.compare_digest(mac.hexdigest(), signature)


# ----------------------------
# Fetch Changed Files (Better than raw diff)
# ----------------------------
def fetch_pr_files(repo_full_name: str, pr_number: int):
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            logger.error(
                f"GitHub API error {response.status_code}: {response.text}"
            )
            return None

        return response.json()

    except requests.RequestException as e:
        logger.error(f"GitHub request failed: {e}")
        return None


# ----------------------------
# Background PR Processing
# ----------------------------
def process_pr(payload: dict):
    try:
        action = payload.get("action")

        if action not in ["opened", "synchronize", "reopened"]:
            logger.info(f"Ignored action: {action}")
            return

        repo = payload["repository"]["full_name"]
        pr_number = payload["pull_request"]["number"]

        logger.info(f"Processing PR #{pr_number} in {repo}")

        files = fetch_pr_files(repo, pr_number)

        if not files:
            logger.warning("No files returned")
            return

        logger.info(f"Files changed: {len(files)}")

        for file in files:
            filename = file.get("filename")
            additions = file.get("additions")
            deletions = file.get("deletions")
            patch = file.get("patch", "")

            logger.info(
                f"File: {filename} (+{additions} -{deletions})"
            )

            if patch:
                logger.info(f"Patch preview:\n{patch[:500]}")

        logger.info("PR processing completed successfully")

    except Exception as e:
        logger.exception(f"Error in process_pr: {e}")


# ----------------------------
# Webhook Endpoint
# ----------------------------
@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):

    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")

    # Verify GitHub signature
    if not verify_signature(body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.info("Webhook triggered")

    background_tasks.add_task(process_pr, payload)

    return {"status": "processing"}


# ----------------------------
# Railway Entry Point
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
