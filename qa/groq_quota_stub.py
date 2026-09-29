from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/chat/completions")
async def quota_exhausted() -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": (
                    "Quota exhausted. Limit 100, Used 100, Requested 1. Please try again in 1s"
                )
            }
        },
        status_code=429,
        headers={"retry-after": "1", "x-request-id": "qa-quota-exhausted"},
    )
