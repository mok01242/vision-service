from fastapi import FastAPI

app = FastAPI(title="Vision Service (smoke test)")

@app.get("/health")
def health():
    return {"ok": True}
