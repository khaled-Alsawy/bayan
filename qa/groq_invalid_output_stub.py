from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/chat/completions")
async def invalid_completion() -> JSONResponse:
    return JSONResponse(
        {
            "id": "qa-invalid-output",
            "model": "openai/gpt-oss-20b",
            "choices": [{"message": {"content": "not valid structured JSON"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        headers={"x-request-id": "qa-invalid-output"},
    )
