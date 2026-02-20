import os
import logging
from fastapi import FastAPI, Request
import uvicorn

# --------------------
# Logging
# --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


# --------------------
# Health Check
# --------------------
@app.get("/")
def health():
    return {"status": "running"}


# --------------------
# Simple Webhook
# --------------------
@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return {"error": "invalid json"}

    action = payload.get("action")

    logger.info("Webhook received")
    logger.info(f"Action: {action}")

    if "pull_request" in payload:
        pr_number = payload["pull_request"]["number"]
        repo = payload["repository"]["full_name"]
        logger.info(f"PR #{pr_number} in {repo}")

    return {"status": "received"}


# --------------------
# Railway Entry
# --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
