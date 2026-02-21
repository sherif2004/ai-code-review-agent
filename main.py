import os
import logging
import hashlib
import hmac
import time
import requests
from collections import deque
from datetime import datetime, timezone
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# =========================================================
# CONFIG
# =========================================================

GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TEST_MODE          = os.getenv("TEST_MODE", "false").lower() == "true"
WEBHOOK_SECRET     = os.getenv("WEBHOOK_SECRET", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

LLM_TIMEOUT_SECONDS    = int(os.getenv("LLM_TIMEOUT", "60"))
GITHUB_TIMEOUT_SECONDS = 15
MAX_HISTORY            = 50   # keep last N reviews in memory

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
# IN-MEMORY REVIEW HISTORY
# =========================================================

review_history: deque = deque(maxlen=MAX_HISTORY)

def add_to_history(repo: str, pr_number: int, action: str, review: str, duration_s: float):
    review_history.appendleft({
        "id": f"{repo}#{pr_number}-{int(time.time())}",
        "repo": repo,
        "pr_number": pr_number,
        "action": action,
        "review": review,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(duration_s, 2),
        "status": "error" if review.startswith("❌") else "warning" if review.startswith("⚠️") else "ok",
    })

# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "running",
        "test_mode": TEST_MODE,
        "model": OPENROUTER_MODEL,
        "github_token_set": bool(GITHUB_TOKEN),
        "openrouter_key_set": bool(OPENROUTER_API_KEY),
        "review_count": len(review_history),
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
        "(bugs, security issues, style, performance). Use markdown with headers and bullet points:\n\n"
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

def fetch_pr_meta(repo_full_name: str, pr_number: int) -> dict | None:
    """Fetch PR title, author, and description."""
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    try:
        response = requests.get(url, headers=_github_headers(), timeout=GITHUB_TIMEOUT_SECONDS)
        if response.status_code == 200:
            d = response.json()
            return {
                "title": d.get("title", ""),
                "author": d.get("user", {}).get("login", ""),
                "body": d.get("body", ""),
                "base": d.get("base", {}).get("ref", ""),
                "head": d.get("head", {}).get("ref", ""),
                "changed_files": d.get("changed_files", 0),
                "additions": d.get("additions", 0),
                "deletions": d.get("deletions", 0),
                "html_url": d.get("html_url", ""),
            }
    except Exception:
        pass
    return None

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
# CORE PR PROCESSING
# =========================================================

def build_patch_from_files(files: list) -> str:
    combined = ""
    for f in files:
        patch = f.get("patch")
        if patch:
            combined += f"\n\n### File: {f['filename']}\n{patch}"
    return combined

def review_pr(repo: str, pr_number: int) -> tuple[str, dict | None]:
    """Returns (review_text, pr_meta_or_None)."""
    meta = fetch_pr_meta(repo, pr_number)
    files = fetch_pr_files(repo, pr_number)
    if files is None:
        return "❌ Could not fetch PR files from GitHub.", meta

    patch = build_patch_from_files(files)
    if not patch:
        return "⚠️ No patchable content found in this PR (maybe only binary files).", meta

    logger.info("Patch built: %d chars for PR #%d", len(patch), pr_number)
    return analyze_code_with_llm(patch), meta

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

    t0 = time.time()
    review, meta = review_pr(repo, pr_number)
    duration = time.time() - t0

    add_to_history(repo, pr_number, action, review, duration)
    logger.info("Posting review comment to PR #%d...", pr_number)
    post_pr_comment(repo, pr_number, review)
    logger.info("Background processing complete for PR #%d", pr_number)

# =========================================================
# WEBHOOK SECRET VALIDATION
# =========================================================

def verify_signature(body: bytes, signature_header: str | None) -> bool:
    if not WEBHOOK_SECRET:
        return True
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

    if not body:
        raise HTTPException(status_code=400, detail="Empty request body.")

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
        t0 = time.time()
        review, meta = review_pr(repo, pr_number)
        duration = time.time() - t0
        add_to_history(repo, pr_number, action or "debug", review, duration)
        return {"status": "debug_complete", "pr": pr_number, "llm_review": review, "meta": meta}

    background_tasks.add_task(process_pr, payload)
    return {"status": "processing", "pr": pr_number, "action": action}

# =========================================================
# API: MANUAL REVIEW + HISTORY
# =========================================================

@app.get("/api/review")
def api_review(repo: str, pr: int, post_comment: bool = False):
    """
    Manually trigger a review. Optionally post the comment back to GitHub.
    Example: /api/review?repo=user/repo&pr=4&post_comment=true
    """
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="GITHUB_TOKEN not set.")
    t0 = time.time()
    review, meta = review_pr(repo, pr)
    duration = time.time() - t0
    add_to_history(repo, pr, "manual", review, duration)

    if post_comment and not review.startswith("❌"):
        posted = post_pr_comment(repo, pr, review)
    else:
        posted = False

    return {
        "repo": repo,
        "pr": pr,
        "meta": meta,
        "llm_review": review,
        "comment_posted": posted,
        "duration_s": round(duration, 2),
    }

@app.get("/api/history")
def api_history():
    return {"reviews": list(review_history)}

@app.get("/api/status")
def api_status():
    return {
        "status": "running",
        "test_mode": TEST_MODE,
        "model": OPENROUTER_MODEL,
        "github_token_set": bool(GITHUB_TOKEN),
        "openrouter_key_set": bool(OPENROUTER_API_KEY),
        "review_count": len(review_history),
        "llm_timeout": LLM_TIMEOUT_SECONDS,
    }

@app.get("/api/test-llm")
def api_test_llm():
    sample = "def add(a, b): return a+b\n\npassword = 'hunter2'  # TODO: fix"
    t0 = time.time()
    result = analyze_code_with_llm(sample)
    return {"llm_response": result, "duration_s": round(time.time() - t0, 2)}

# =========================================================
# UI
# =========================================================

@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(content=HTML_PAGE)

# =========================================================
# HTML UI (single-file, no build step)
# =========================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Code Review Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0c0f;
    --surface: #111418;
    --surface2: #181c22;
    --border: #1f2530;
    --accent: #00e5a0;
    --accent2: #0088ff;
    --warn: #ffb800;
    --err: #ff4757;
    --text: #dde3ed;
    --muted: #5a6478;
    --card-shadow: 0 2px 24px rgba(0,0,0,0.5);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
    min-height: 100vh;
  }

  /* ---- top bar ---- */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 32px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 18px;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .logo-dot {
    width: 10px; height: 10px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 12px var(--accent);
    animation: pulse 2s infinite;
  }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .status-pills { display: flex; gap: 8px; }

  .pill {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--muted);
    transition: all .2s;
  }
  .pill.ok   { border-color: var(--accent); color: var(--accent); }
  .pill.err  { border-color: var(--err); color: var(--err); }
  .pill.warn { border-color: var(--warn); color: var(--warn); }

  /* ---- layout ---- */
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 24px 80px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }

  .full { grid-column: 1 / -1; }

  /* ---- card ---- */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px;
    box-shadow: var(--card-shadow);
    animation: fadeUp .4s ease both;
  }

  @keyframes fadeUp {
    from { opacity:0; transform: translateY(12px); }
    to   { opacity:1; transform: translateY(0); }
  }

  .card:nth-child(2) { animation-delay: .05s; }
  .card:nth-child(3) { animation-delay: .10s; }
  .card:nth-child(4) { animation-delay: .15s; }

  .card-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: -.3px;
    color: #fff;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .card-title .icon { font-size: 16px; }

  /* ---- form elements ---- */
  label {
    display: block;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: 6px;
  }

  input[type=text], input[type=number] {
    width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    margin-bottom: 14px;
    transition: border-color .2s;
    outline: none;
  }

  input:focus { border-color: var(--accent); }

  .row { display: flex; gap: 12px; }
  .row > * { flex: 1; }

  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    color: var(--muted);
    cursor: pointer;
    user-select: none;
  }

  .checkbox-row input { width: auto; margin: 0; accent-color: var(--accent); }

  /* ---- buttons ---- */
  .btn {
    width: 100%;
    padding: 12px 20px;
    border-radius: 6px;
    border: none;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .5px;
    cursor: pointer;
    transition: all .15s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }

  .btn-primary {
    background: var(--accent);
    color: #000;
  }

  .btn-primary:hover:not(:disabled) {
    background: #00ffb3;
    box-shadow: 0 0 24px rgba(0,229,160,.35);
    transform: translateY(-1px);
  }

  .btn-secondary {
    background: transparent;
    color: var(--accent2);
    border: 1px solid var(--accent2);
  }

  .btn-secondary:hover:not(:disabled) {
    background: rgba(0,136,255,.1);
    box-shadow: 0 0 18px rgba(0,136,255,.2);
  }

  .btn:disabled { opacity: .4; cursor: not-allowed; }

  /* ---- spinner ---- */
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid rgba(0,0,0,.3);
    border-top-color: #000;
    border-radius: 50%;
    animation: spin .6s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---- pr meta bar ---- */
  .pr-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 16px;
    padding: 12px 16px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 12px;
    color: var(--muted);
  }

  .pr-meta span { display: flex; align-items: center; gap: 4px; }
  .pr-meta a { color: var(--accent2); text-decoration: none; }
  .pr-meta a:hover { text-decoration: underline; }

  .stat-add { color: var(--accent); }
  .stat-del { color: var(--err); }

  /* ---- review output ---- */
  .output-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    min-height: 120px;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 13px;
    line-height: 1.8;
    display: none;
    animation: fadeUp .3s ease;
  }

  .output-box.visible { display: block; }

  /* Basic markdown rendering inside output-box */
  .output-box h1, .output-box h2, .output-box h3 {
    font-family: 'Syne', sans-serif;
    color: #fff;
    margin: 16px 0 8px;
  }
  .output-box h3 { font-size: 14px; color: var(--accent); }
  .output-box h2 { font-size: 15px; }
  .output-box ul { padding-left: 18px; }
  .output-box li { margin-bottom: 4px; }
  .output-box code {
    background: rgba(0,229,160,.08);
    border: 1px solid rgba(0,229,160,.15);
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 12px;
    color: var(--accent);
  }
  .output-box pre {
    background: rgba(0,0,0,.35);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
    margin: 10px 0;
  }
  .output-box pre code {
    background: none; border: none; padding: 0; color: var(--text);
  }
  .output-box strong { color: #fff; }

  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-bottom: 12px;
  }
  .badge-ok   { background: rgba(0,229,160,.12); color: var(--accent); border: 1px solid rgba(0,229,160,.25); }
  .badge-err  { background: rgba(255,71,87,.12);  color: var(--err);   border: 1px solid rgba(255,71,87,.25); }
  .badge-warn { background: rgba(255,184,0,.12);  color: var(--warn);  border: 1px solid rgba(255,184,0,.25); }

  .duration { float: right; color: var(--muted); font-size: 11px; }

  /* ---- history table ---- */
  .history-table { width: 100%; border-collapse: collapse; }

  .history-table th {
    text-align: left;
    padding: 8px 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .7px;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
  }

  .history-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    vertical-align: middle;
  }

  .history-table tr:last-child td { border-bottom: none; }

  .history-table tr:hover td { background: rgba(255,255,255,.02); }

  .history-table .repo-link { color: var(--accent2); text-decoration: none; }
  .history-table .repo-link:hover { text-decoration: underline; }

  .expand-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--muted);
    cursor: pointer;
    padding: 2px 8px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    transition: all .15s;
  }
  .expand-btn:hover { border-color: var(--accent); color: var(--accent); }

  .expanded-review {
    padding: 16px;
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    margin: 4px 0;
    white-space: pre-wrap;
    font-size: 12px;
    line-height: 1.7;
    display: none;
  }
  .expanded-review.open { display: block; }

  /* ---- system status grid ---- */
  .status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-top: 4px;
  }

  .stat-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }

  .stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: var(--muted);
    margin-bottom: 4px;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    color: #fff;
  }

  .stat-value.green { color: var(--accent); }
  .stat-value.red   { color: var(--err); }
  .stat-value.blue  { color: var(--accent2); }

  /* ---- empty state ---- */
  .empty {
    text-align: center;
    padding: 32px;
    color: var(--muted);
    font-size: 12px;
  }

  /* ---- scrollbar ---- */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: var(--surface); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot"></div>
    Code Review Agent
  </div>
  <div class="status-pills" id="headerPills">
    <span class="pill" id="pillGH">GH —</span>
    <span class="pill" id="pillLLM">LLM —</span>
    <span class="pill" id="pillMode">—</span>
  </div>
</header>

<main>

  <!-- ── Review Form ── -->
  <div class="card">
    <div class="card-title"><span class="icon">🔍</span> Review a Pull Request</div>

    <label>Repository (owner/repo)</label>
    <input type="text" id="repoInput" placeholder="e.g. torvalds/linux" />

    <label>PR Number</label>
    <input type="number" id="prInput" placeholder="e.g. 42" />

    <label class="checkbox-row">
      <input type="checkbox" id="postComment" />
      Post review as GitHub comment after analysis
    </label>

    <button class="btn btn-primary" id="reviewBtn" onclick="runReview()">
      <div class="spinner" id="reviewSpinner"></div>
      <span id="reviewBtnText">▶ Run Review</span>
    </button>
  </div>

  <!-- ── System Status ── -->
  <div class="card">
    <div class="card-title"><span class="icon">⚡</span> System Status</div>
    <div class="status-grid" id="statusGrid">
      <div class="stat-item">
        <div class="stat-label">Model</div>
        <div class="stat-value blue" id="sModel">—</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">Reviews Run</div>
        <div class="stat-value" id="sCount">—</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">GitHub Token</div>
        <div class="stat-value" id="sGH">—</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">LLM Key</div>
        <div class="stat-value" id="sLLM">—</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">LLM Timeout</div>
        <div class="stat-value" id="sTimeout">—</div>
      </div>
      <div class="stat-item">
        <div class="stat-label">Test Mode</div>
        <div class="stat-value" id="sTest">—</div>
      </div>
    </div>
    <br/>
    <button class="btn btn-secondary" onclick="testLLM()" id="testLLMBtn">
      <div class="spinner" id="llmSpinner" style="border-top-color: var(--accent2);"></div>
      <span id="testBtnText">🧪 Smoke-test LLM</span>
    </button>
  </div>

  <!-- ── Review Output ── -->
  <div class="card full" id="reviewOutputCard" style="display:none">
    <div class="card-title"><span class="icon">📋</span> Review Output</div>
    <div id="prMetaBar" class="pr-meta" style="display:none"></div>
    <div id="reviewStatusBadge"></div>
    <div class="output-box visible" id="reviewOutput"></div>
  </div>

  <!-- ── History ── -->
  <div class="card full">
    <div class="card-title">
      <span class="icon">🕓</span> Review History
      <button class="expand-btn" style="margin-left:auto" onclick="loadHistory()">↻ Refresh</button>
    </div>
    <div id="historyContainer">
      <div class="empty">No reviews yet this session.</div>
    </div>
  </div>

</main>

<script>
// ── Marked.js lite for basic markdown ──
function renderMarkdown(text) {
  return text
    .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

// ── Status ──
async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    document.getElementById('sModel').textContent   = d.model.split('/').pop();
    document.getElementById('sCount').textContent   = d.review_count;
    document.getElementById('sTimeout').textContent = d.llm_timeout + 's';
    document.getElementById('sTest').textContent    = d.test_mode ? 'ON' : 'OFF';
    document.getElementById('sTest').className      = 'stat-value ' + (d.test_mode ? 'red' : 'green');

    const ghOk = d.github_token_set;
    const llmOk = d.openrouter_key_set;
    document.getElementById('sGH').textContent  = ghOk  ? '✓ Set' : '✗ Missing';
    document.getElementById('sGH').className    = 'stat-value ' + (ghOk  ? 'green' : 'red');
    document.getElementById('sLLM').textContent = llmOk ? '✓ Set' : '✗ Missing';
    document.getElementById('sLLM').className   = 'stat-value ' + (llmOk ? 'green' : 'red');

    // header pills
    const pillGH = document.getElementById('pillGH');
    pillGH.textContent = 'GitHub';
    pillGH.className   = 'pill ' + (ghOk  ? 'ok' : 'err');

    const pillLLM = document.getElementById('pillLLM');
    pillLLM.textContent = 'LLM';
    pillLLM.className   = 'pill ' + (llmOk ? 'ok' : 'err');

    const pillMode = document.getElementById('pillMode');
    pillMode.textContent = d.test_mode ? 'TEST MODE' : 'LIVE';
    pillMode.className   = 'pill ' + (d.test_mode ? 'warn' : 'ok');

  } catch(e) {
    console.error('Status fetch failed', e);
  }
}

// ── Review ──
async function runReview() {
  const repo = document.getElementById('repoInput').value.trim();
  const pr   = document.getElementById('prInput').value.trim();
  const post = document.getElementById('postComment').checked;

  if (!repo || !pr) { alert('Please enter both repository and PR number.'); return; }

  setLoading('reviewBtn', 'reviewSpinner', 'reviewBtnText', true, 'Reviewing…');

  document.getElementById('reviewOutputCard').style.display = 'none';

  try {
    const url = `/api/review?repo=${encodeURIComponent(repo)}&pr=${pr}&post_comment=${post}`;
    const res = await fetch(url);
    const data = await res.json();

    // Show output card
    const card = document.getElementById('reviewOutputCard');
    card.style.display = 'block';

    // PR meta bar
    const metaBar = document.getElementById('prMetaBar');
    if (data.meta) {
      const m = data.meta;
      metaBar.style.display = 'flex';
      metaBar.innerHTML = `
        <span>📄 <a href="${m.html_url}" target="_blank">${m.title || 'PR #' + pr}</a></span>
        <span>👤 ${m.author}</span>
        <span>${m.base} ← ${m.head}</span>
        <span>📁 ${m.changed_files} file${m.changed_files!=1?'s':''}</span>
        <span class="stat-add">+${m.additions}</span>
        <span class="stat-del">-${m.deletions}</span>
        ${data.comment_posted ? '<span>💬 Comment posted ✓</span>' : ''}
      `;
    } else {
      metaBar.style.display = 'none';
    }

    // Status badge
    const review = data.llm_review || '';
    let badgeClass = 'badge-ok', badgeText = '✓ Review Complete';
    if (review.startsWith('❌')) { badgeClass = 'badge-err'; badgeText = '✗ Error'; }
    else if (review.startsWith('⚠️')) { badgeClass = 'badge-warn'; badgeText = '⚠ Warning'; }
    document.getElementById('reviewStatusBadge').innerHTML =
      `<div class="status-badge ${badgeClass}">${badgeText}</div>` +
      `<span class="duration">${data.duration_s}s</span>`;

    // Review text
    const box = document.getElementById('reviewOutput');
    box.innerHTML = renderMarkdown(review);

    loadHistory();
    loadStatus();
  } catch(e) {
    alert('Request failed: ' + e.message);
  } finally {
    setLoading('reviewBtn', 'reviewSpinner', 'reviewBtnText', false, '▶ Run Review');
  }
}

// ── Smoke test LLM ──
async function testLLM() {
  setLoading('testLLMBtn', 'llmSpinner', 'testBtnText', true, 'Testing…');
  try {
    const res  = await fetch('/api/test-llm');
    const data = await res.json();

    const card = document.getElementById('reviewOutputCard');
    card.style.display = 'block';
    document.getElementById('prMetaBar').style.display = 'none';
    document.getElementById('reviewStatusBadge').innerHTML =
      `<div class="status-badge badge-ok">🧪 LLM Smoke Test — ${data.duration_s}s</div>`;
    document.getElementById('reviewOutput').innerHTML = renderMarkdown(data.llm_response);
  } catch(e) {
    alert('Test failed: ' + e.message);
  } finally {
    setLoading('testLLMBtn', 'llmSpinner', 'testBtnText', false, '🧪 Smoke-test LLM');
  }
}

// ── History ──
async function loadHistory() {
  try {
    const res  = await fetch('/api/history');
    const data = await res.json();
    const reviews = data.reviews || [];

    const container = document.getElementById('historyContainer');
    if (reviews.length === 0) {
      container.innerHTML = '<div class="empty">No reviews yet this session.</div>';
      return;
    }

    const badgeFor = s => {
      if (s === 'ok')      return '<span class="status-badge badge-ok" style="padding:1px 6px;font-size:10px">OK</span>';
      if (s === 'error')   return '<span class="status-badge badge-err" style="padding:1px 6px;font-size:10px">ERR</span>';
      return '<span class="status-badge badge-warn" style="padding:1px 6px;font-size:10px">WARN</span>';
    };

    const rows = reviews.map((r, i) => {
      const ts = new Date(r.timestamp).toLocaleString();
      return `
        <tr>
          <td>${badgeFor(r.status)}</td>
          <td><a class="repo-link" href="https://github.com/${r.repo}/pull/${r.pr_number}" target="_blank">${r.repo} #${r.pr_number}</a></td>
          <td>${r.action}</td>
          <td>${ts}</td>
          <td>${r.duration_s}s</td>
          <td><button class="expand-btn" onclick="toggleExpand(${i})">view</button></td>
        </tr>
        <tr id="exp-${i}">
          <td colspan="6">
            <div class="expanded-review" id="expContent-${i}">${renderMarkdown(r.review)}</div>
          </td>
        </tr>
      `;
    }).join('');

    container.innerHTML = `
      <table class="history-table">
        <thead><tr>
          <th>Status</th><th>PR</th><th>Action</th><th>Time</th><th>Duration</th><th></th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch(e) {
    console.error('History fetch failed', e);
  }
}

function toggleExpand(i) {
  const el = document.getElementById('expContent-' + i);
  el.classList.toggle('open');
}

// ── Helpers ──
function setLoading(btnId, spinnerId, textId, loading, text) {
  const btn = document.getElementById(btnId);
  btn.disabled = loading;
  document.getElementById(spinnerId).style.display = loading ? 'block' : 'none';
  document.getElementById(textId).textContent = text;
}

// ── Init ──
loadStatus();
loadHistory();
setInterval(loadStatus, 30000);
</script>
</body>
</html>
"""

# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
