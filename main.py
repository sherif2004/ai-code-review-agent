import os
import logging
from fastapi import FastAPI, Request
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/webhook")
async def webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        logger.info("Invalid JSON received")
        return {"error": "invalid json"}

    logger.info("Webhook received")

    action = payload.get("action")
    logger.info(f"Action: {action}")

    if "pull_request" in payload:
        pr_number = payload["pull_request"]["number"]
        repo = payload["repository"]["full_name"]
        logger.info(f"PR #{pr_number} in {repo}")

    return {"status": "received"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
