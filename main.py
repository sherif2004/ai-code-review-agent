import os
import hashlib
import hmac
import logging
import requests
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
import uvicorn

# =========================================================
# CONFIG
# =========================================================

GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WEBHOOK_SECRET     = os.getenv("WEBHOOK_SECRET", "")        # Optional but recommended
TEST_MODE          = os.getenv("TEST_MODE", "false").lower() == "true"

OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "mistralai/mistral-7b-instruct"

LLM_TIMEOUT_SECONDS    = 60
GITHUB_TIMEOUT_SECONDS = 15

ALLOWED_ACTIONS = {"opened", "synchronize", "reopened"}

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Code Review Agent")

# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def health():
    return {
        "status":             "running",
        "test_mode":          TEST_MODE,
        "model":              OPENROUTER_MODEL,
        "github_token_set":   bool(GITHUB_TOKEN),
        "openrouter_key_set": bool(OPENROUTER_API_KEY),
        "webhook_secret_set": bool(WEBHOOK_SECRET),
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

    logger.info(
        "[llm] Sending request | model=%s | prompt_chars=%d",
        OPENROUTER_MODEL, len(prompt),
    )

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,      # ✅ FIX: was missing — caused silent hangs
        )
    except requests.Timeout:
        logger.error("[llm] Request timed out after %ds", LLM_TIMEOUT_SECONDS)
        return f"❌ LLM request timed out after {LLM_TIMEOUT_SECONDS}s."
    except requests.RequestException as e:
        logger.error("[llm] Connection error: %s", e)
        return f"❌ LLM connection error: {e}"

    logger.info("[llm] Response status: %d", response.status_code)

    if response.status_code != 200:
        logger.error("[llm] Error body: %s", response.text)
        return f"❌ LLM call failed (HTTP {response.status_code}): {response.text}"

    try:
        output = response.json()["choices"][0]["message"]["content"]
        logger.info("[llm] Response received successfully (%d chars)", len(output))
        return output
    except (KeyError, IndexError) as e:
        logger.error("[llm] Unexpected response format: %s | raw: %s", e, response.text)
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
        logger.error("[github] GITHUB_TOKEN is not set")
        return None

    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"

    try:
        response = requests.get(
            url,
            headers=_github_headers(),
            timeout=GITHUB_TIMEOUT_SECONDS,   # ✅ FIX: was missing
        )
    except requests.RequestException as e:
        logger.error("[github] Connection error fetching PR files: %s", e)
        return None

    if response.status_code != 200:
        logger.error(
            "[github] API error (HTTP %d): %s",
            response.status_code, response.text,
        )
        return None

    files = response.json()
    logger.info("[github] Fetched %d file(s) from PR #%d", len(files), pr_number)
    return files

def post_pr_comment(repo_full_name: str, pr_number: int, comment: str) -> bool:
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"

    try:
        response = requests.post(
            url,
            headers=_github_headers(),
            json={"body": comment},
            timeout=GITHUB_TIMEOUT_SECONDS,   # ✅ FIX: was missing
        )
    except requests.RequestException as e:
        logger.error("[github] Connection error posting comment: %s", e)
        return False

    if response.status_code != 201:
        logger.error(
            "[github] Failed to post comment (HTTP %d): %s",
            response.status_code, response.text,
        )
        return False

    logger.info("[github] Comment posted to PR #%d successfully", pr_number)
    return True

# =========================================================
# CORE REVIEW LOGIC  (shared by /webhook background task + /test-pr)
# =========================================================

def build_patch_from_files(files: list) -> str:
    """Combine all file patches into one diff string."""
    combined = ""
    for f in files:
        patch = f.get("patch")
        if patch:
            combined += f"\n\n### File: {f['filename']}\n{patch}"
    return combined

def review_pr(repo: str, pr_number: int) -> str:
    """
    Fetch PR files → build patch → ask LLM → return review text.
    Single source of truth — used by both the webhook and /test-pr.
    """
    files = fetch_pr_files(repo, pr_number)
    if not files:
        return "❌ Could not fetch PR files from GitHub."

    patch = build_patch_from_files(files)
    if not patch:
        return "⚠️ No patchable content found in this PR (maybe only binary files)."

    logger.info(
        "[review] Patch built: %d chars | repo=%s | PR=#%d",
        len(patch), repo, pr_number,
    )
    return analyze_code_with_llm(patch)

# =========================================================
# BACKGROUND WRAPPER
# =========================================================

def process_pr(payload: dict):
    """
    Background task. Wraps review_pr() with:
      - Defensive field extraction with individual warnings
      - Action allow-list filtering
      - Full exception capture — errors are NEVER silent
    """
    try:
        # ── Defensive extraction ──────────────────────────────────────
        action = payload.get("action")
        if not action:
            logger.warning("[process_pr] Missing 'action' field — skipping")
            return

        pull_request = payload.get("pull_request")
        if not pull_request:
            logger.warning("[process_pr] Missing 'pull_request' field — skipping")
            return

        repository = payload.get("repository")
        if not repository or not repository.get("full_name"):
            logger.warning("[process_pr] Missing 'repository.full_name' — skipping")
            return

        repo      = repository["full_name"]
        pr_number = pull_request.get("number")

        if not pr_number:
            logger.warning("[process_pr] Missing 'pull_request.number' — skipping")
            return

        # ── Action guard ──────────────────────────────────────────────
        if action not in ALLOWED_ACTIONS:
            logger.info(
                "[process_pr] Skipped | repo=%s | PR=#%d | action=%s (not in allowed set)",
                repo, pr_number, action,
            )
            return

        logger.info(
            "[process_pr] Starting | repo=%s | PR=#%d | action=%s",
            repo, pr_number, action,
        )

        # ── TEST_MODE guard ───────────────────────────────────────────
        if TEST_MODE:
            logger.info("[process_pr] TEST_MODE=true — skipping LLM and comment posting")
            return

        # ── Run review and post comment ───────────────────────────────
        review = review_pr(repo, pr_number)
        posted = post_pr_comment(repo, pr_number, review)

        if posted:
            logger.info(
                "[process_pr] Done | repo=%s | PR=#%d | comment posted ✓",
                repo, pr_number,
            )
        else:
            logger.error(
                "[process_pr] Review generated but comment post failed | repo=%s | PR=#%d",
                repo, pr_number,
            )

    except Exception:
        # logger.exception always includes the full traceback
        logger.exception("[process_pr] Unhandled exception during background processing")

# =========================================================
# WEBHOOK SIGNATURE VALIDATION
# =========================================================

def verify_github_signature(body: bytes, signature_header: str | None) -> bool:
    """
    Returns True  → validation passed (or WEBHOOK_SECRET not configured → skip).
    Returns False → bad/missing signature → caller returns 401.
    """
    if not WEBHOOK_SECRET:
        logger.info("[webhook] WEBHOOK_SECRET not set — skipping signature validation")
        return True

    if not signature_header:
        logger.warning("[webhook] Signature FAILED — X-Hub-Signature-256 header missing")
        return False

    if not signature_header.startswith("sha256="):
        logger.warning(
            "[webhook] Signature FAILED — unexpected format: %s",
            signature_header[:20],
        )
        return False

    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()

    valid = hmac.compare_digest(expected, signature_header)

    if valid:
        logger.info("[webhook] Signature PASSED ✓")
    else:
        logger.warning(
            "[webhook] Signature FAILED — received=%s | expected=%s",
            signature_header[:20], expected[:20],
        )

    return valid

# =========================================================
# WEBHOOK ENDPOINT  (thin HTTP layer — never blocks on LLM)
# =========================================================

@app.post("/webhook", status_code=200)
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Thin HTTP layer only:
      1. Read raw body
      2. Validate GitHub signature
      3. Parse JSON
      4. Defensive field checks + structured logging
      5. Ignore non-PR events → return immediately
      6. Queue background task → return 200 immediately
         (GitHub must get a response before LLM finishes)
    """

    # ── 1. Read body ──────────────────────────────────────────────────
    body = await request.body()
    if not body:
        logger.warning("[webhook] Empty body received — returning 400")
        raise HTTPException(
            status_code=400,
            detail="Empty request body. Expecting a GitHub webhook payload.",
        )

    # ── 2. Signature validation ───────────────────────────────────────
    signature_header = request.headers.get("X-Hub-Signature-256")
    if not verify_github_signature(body, signature_header):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing webhook signature.",
        )

    # ── 3. Parse JSON ─────────────────────────────────────────────────
    try:
        payload = await request.json()
    except Exception:
        logger.warning("[webhook] Failed to parse JSON body")
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    # ── 4. Defensive field extraction + structured logging ────────────
    action     = payload.get("action")
    repository = payload.get("repository") or {}
    repo       = repository.get("full_name", "unknown")
    pr_data    = payload.get("pull_request") or {}
    pr_number  = pr_data.get("number", "unknown")

    logger.info(
        "[webhook] Event received | repo=%s | PR=#%s | action=%s | has_pull_request=%s",
        repo,
        pr_number,
        action,
        "pull_request" in payload,
    )

    # ── 5. Ignore non-PR events ───────────────────────────────────────
    if "pull_request" not in payload:
        logger.info("[webhook] Ignored — not a pull_request event | action=%s", action)
        return {"status": "ignored", "reason": "not a pull_request event"}

    # ── 6. Log action detail and queue background task ────────────────
    logger.info(
        "[webhook] PR event | repo=%s | PR=#%s | action=%s | will_process=%s",
        repo,
        pr_number,
        action,
        action in ALLOWED_ACTIONS,
    )

    background_tasks.add_task(process_pr, payload)
    logger.info("[webhook] Background task queued | repo=%s | PR=#%s", repo, pr_number)

    return {
        "status": "queued",
        "repo":   repo,
        "pr":     pr_number,
        "action": action,
    }

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
    Manually trigger a synchronous review for any PR.
    Used by the dashboard UI and for direct debugging.
    Example: /test-pr?repo=sherif2004/PSO_TSP&pr=4
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
