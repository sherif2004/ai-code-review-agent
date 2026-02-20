from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/webhook")
async def webhook():
    return {"received": True}