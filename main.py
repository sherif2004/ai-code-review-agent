import os
import logging
import hashlib
import hmac
import requests
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
import uvicorn

# =========================================================
# CONFIG
# =========================================================

GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TEST_MODE          = os.getenv("TEST_MODE", "false").lower() == "true"
WEBHOOK_SECRET     = os.getenv("WEBHOOK_SECRET", "")       # Optional but recommended
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "mistralai/mistral-7b-instruct"

LLM_TIMEOUT_SECONDS    = 60   # ✅ FIX: was missing entirely — caused silent hangs
GITHUB_TIMEOUT_SECONDS = 15

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Code Review Agent")

# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def health():
    return {
        "status": "running",
        "test_mode": TEST_MODE,
        "model": OPENROUTER_MODEL,
        "github_token_set": bool(GITHUB_TOKEN),
        "openrouter_key_set": bool(OPENROUTER_API_KEY),
    }

# =========================================================
# OPENROUTER LLM CALL
# =========================================================

def analyze_code_with_llm(text: str) -> str:
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set")
        return "❌ LLM key missing — set OPENROUTER_API_KEY in your Railway environment."

    prompt = (
        "You are a senior software engineer.\n\n"
        "Review the following code diff and give clear, actionable improvement suggestions "
        "(bugs, security issues, style, performance):\n\n"
        f"{text}"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://railway.app",
        "X-Title": "AI Code Review Agent",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }

    logger.info("Sending request to OpenRouter | model=%s | prompt_chars=%d", OPENROUTER_MODEL, len(prompt))

    try:
        # ✅ FIX: timeout added — this was the root cause of silent hangs
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        logger.error("OpenRouter request timed out after %ds", LLM_TIMEOUT_SECONDS)
        return f"❌ LLM request timed out after {LLM_TIMEOUT_SECONDS}s."
    except requests.RequestException as e:
        logger.error("OpenRouter connection error: %s", e)
        return f"❌ LLM connection error: {e}"

    logger.info("OpenRouter response status: %d", response.status_code)

    if response.status_code != 200:
        logger.error("OpenRouter error body: %s", response.text)
        return f"❌ LLM call failed (HTTP {response.status_code}): {response.text}"

    try:
        output = response.json()["choices"][0]["message"]["content"]
        logger.info("LLM response received successfully (%d chars)", len(output))
        return output
    except (KeyError, IndexError) as e:
        logger.error("Unexpected OpenRouter response format: %s | raw: %s", e, response.text)
        return "❌ LLM returned an unexpected response format."

# =========================================================
# GITHUB HELPERS
# =========================================================

def _github_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def fetch_pr_files(repo_full_name: str, pr_number: int) -> list | None:
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN is not set")
        return None

    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"

    try:
        response = requests.get(url, headers=_github_headers(), timeout=GITHUB_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        logger.error("GitHub API connection error: %s", e)
        return None

    if response.status_code != 200:
        logger.error("GitHub API error (HTTP %d): %s", response.status_code, response.text)
        return None

    files = response.json()
    logger.info("Fetched %d file(s) from PR #%d", len(files), pr_number)
    return files

def post_pr_comment(repo_full_name: str, pr_number: int, comment: str) -> bool:
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"

    try:
        response = requests.post(
            url,
            headers=_github_headers(),
            json={"body": comment},
            timeout=GITHUB_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        logger.error("GitHub comment post connection error: %s", e)
        return False

    if response.status_code != 201:
        logger.error("Failed to post comment (HTTP %d): %s", response.status_code, response.text)
        return False

    logger.info("Comment posted to PR #%d successfully", pr_number)
    return True

# =========================================================
# CORE PR PROCESSING (shared by background + debug mode)
# ✅ FIX: eliminated code duplication between process_pr and run_full_pr_flow
# =========================================================

def build_patch_from_files(files: list) -> str:
    """Combine all file patches into a single diff string."""
    combined = ""
    for f in files:
        patch = f.get("patch")
        if patch:
            combined += f"\n\n### File: {f['filename']}\n{patch}"
    return combined

def review_pr(repo: str, pr_number: int) -> str:
    """
    Fetch PR files, build the patch, and get LLM review.
    Returns the review string (or an error message).
    """
    files = fetch_pr_files(repo, pr_number)
    if not files:
        return "❌ Could not fetch PR files from GitHub."

    patch = build_patch_from_files(files)
    if not patch:
        return "⚠️ No patchable content found in this PR (maybe only binary files)."

    logger.info("Patch built: %d chars for PR #%d", len(patch), pr_number)
    return analyze_code_with_llm(patch)

def process_pr(payload: dict):
    """Background task: process PR webhook and post comment."""
    action = payload.get("action")
    if action not in ("opened", "synchronize", "reopened"):
        logger.info("Skipped webhook action: %s", action)
        return

    repo      = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]
    logger.info("Background processing PR #%d in %s (action=%s)", pr_number, repo, action)

    if TEST_MODE:
        logger.info("TEST_MODE=true — skipping LLM call and comment posting")
        return

    review = review_pr(repo, pr_number)
    logger.info("Posting review comment to PR #%d...", pr_number)
    post_pr_comment(repo, pr_number, review)
    logger.info("Background processing complete for PR #%d", pr_number)

# =========================================================
# WEBHOOK SECRET VALIDATION (optional but recommended)
# =========================================================

def verify_signature(body: bytes, signature_header: str | None) -> bool:
    """Return True if signature matches or no secret is configured."""
    if not WEBHOOK_SECRET:
        return True  # Secret not configured, skip check
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)

# =========================================================
# WEBHOOK ENDPOINT
# =========================================================

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks, debug: bool = False):
    body = await request.body()

    # ✅ FIX: return a proper 400 instead of a silent "ok" on empty body
    if not body:
        raise HTTPException(
            status_code=400,
            detail="Empty request body. Send a real GitHub webhook payload."
        )

    # Validate GitHub webhook signature if secret is set
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(body, signature):
        logger.warning("Invalid webhook signature — request rejected")
        raise HTTPException(status_code=401, detail="Invalid webhook signature.")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    if "pull_request" not in payload:
        return {"status": "ignored", "reason": "not a pull_request event"}

    repo      = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]
    action    = payload.get("action")

    logger.info("Webhook received | repo=%s | PR=#%d | action=%s", repo, pr_number, action)

    if debug:
        # Synchronous path — useful for testing; returns LLM output directly
        logger.info("Debug mode — running synchronously")
        review = review_pr(repo, pr_number)
        return {"status": "debug_complete", "pr": pr_number, "llm_review": review}

    # Normal production path — run in background so GitHub doesn't time out
    background_tasks.add_task(process_pr, payload)
    return {"status": "processing", "pr": pr_number, "action": action}

# =========================================================
# TEST ENDPOINTS
# =========================================================

@app.get("/test-llm")
def test_llm():
    """Smoke-test OpenRouter without involving GitHub."""
    sample = "def add(a, b): return a+b"
    result = analyze_code_with_llm(sample)
    return {"llm_response": result}

@app.get("/test-pr")
def test_pr(repo: str, pr: int):
    """
    Manually trigger a review for any PR.
    Example: /test-pr?repo=youruser/yourrepo&pr=4
    """
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not set.")
    review = review_pr(repo, pr)
    return {"repo": repo, "pr": pr, "llm_review": review}

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

