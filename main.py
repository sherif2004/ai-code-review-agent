from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "kbhbhbgchg"}

@app.post("/weook")
async def webhook():

    return {"reghvghvhgvgceived": True}

