"""FastAPI app entrypoint."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from optiresearch.api import artifacts, benchmarks, claims, jobs, memory, skills, traces

app = FastAPI(title="OptiResearch Agent MVP", version="0.1.0")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": {"type": exc.__class__.__name__, "message": str(exc)}})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(jobs.router)
app.include_router(traces.router)
app.include_router(memory.router)
app.include_router(artifacts.router)
app.include_router(claims.router)
app.include_router(skills.router)
app.include_router(benchmarks.router)
