"""FastAPI service for the revenue copilot.

    uvicorn copilot.api.main:app_factory --factory      # reads the variables in .env.example

Routes
  POST /auth/token                  OAuth2 password flow -> JWT bearer token
  GET  /me                          who am I
  GET  /healthz                     liveness (no auth)
  GET  /v1/metrics                  metric catalog                        viewer
  GET  /v1/metrics/{name}           query one metric                      viewer
  POST /v1/ask                      ask the agent; low confidence answers are auto-flagged   viewer
  POST /v1/answers/{id}/flag        human-in-the-loop: flag an answer for review             viewer
  GET  /v1/reviews                  review queue                          admin
  POST /v1/reviews/{id}/resolve     approve or correct a flagged answer   admin
  GET  /ops/metrics                 Prometheus-format request metrics     admin
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from ..agent.graph import Copilot
from ..metrics.semantic import MetricError, SemanticLayer
from .security import AuthConfig, User, UserStore, build_dependencies, create_token
from .store import Store, redact

log = logging.getLogger("copilot.api")


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskOut(BaseModel):
    answer_id: str
    answer: str | None
    value: float | None
    confidence: float
    flagged: bool
    review_id: str | None = None
    plan: dict | None = None


class FlagIn(BaseModel):
    reason: str = Field(min_length=3, max_length=1000)


class ResolveIn(BaseModel):
    verdict: Literal["approved", "corrected", "rejected"]
    corrected_answer: str | None = None
    note: str | None = Field(default=None, max_length=1000)


class Telemetry:
    """In-process request counters and latency sums, exported in Prometheus text format."""

    def __init__(self):
        self.requests = defaultdict(int)
        self.latency_ms = defaultdict(float)
        self.agent = defaultdict(int)

    def prometheus(self) -> str:
        lines = ["# TYPE http_requests_total counter"]
        for (route, method, code), n in sorted(self.requests.items()):
            lines.append(f'http_requests_total{{route="{route}",method="{method}",status="{code}"}} {n}')
        lines.append("# TYPE http_request_latency_ms_sum counter")
        for (route, method), v in sorted(self.latency_ms.items()):
            lines.append(f'http_request_latency_ms_sum{{route="{route}",method="{method}"}} {v:.3f}')
        lines.append("# TYPE agent_answers_total counter")
        for k, n in sorted(self.agent.items()):
            lines.append(f'agent_answers_total{{outcome="{k}"}} {n}')
        return "\n".join(lines) + "\n"


def create_app(db_path: str | None = None, users: UserStore | None = None, auth: AuthConfig | None = None,
               backend: str | None = None, review_db: str | None = None, threshold: float | None = None) -> FastAPI:
    db_path = db_path or os.environ.get("DUCKDB_PATH", "data/saas.duckdb")
    if not Path(db_path).exists():
        raise RuntimeError(f"{db_path} not found; run `python -m copilot.data.generate` first")
    users = users or UserStore.from_env()
    auth = auth or AuthConfig()
    threshold = threshold if threshold is not None else float(os.environ.get("REVIEW_CONFIDENCE_THRESHOLD", 0.7))
    sl = SemanticLayer.from_path(db_path)
    agent = Copilot(sl, backend=backend)
    store = Store(review_db or os.environ.get("REVIEW_DB", ":memory:"))
    telemetry = Telemetry()
    current_user, require = build_dependencies(auth, users)
    viewer, admin = require("viewer"), require("admin")

    app = FastAPI(title="SaaS Revenue Copilot", version="0.1.0")
    app.state.store, app.state.telemetry, app.state.agent = store, telemetry, agent

    # ------------------------------------------------------------------ observability
    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        rpath = getattr(route, "path", "unmatched")
        telemetry.requests[(rpath, request.method, response.status_code)] += 1
        telemetry.latency_ms[(rpath, request.method)] += ms
        response.headers["x-request-id"] = rid
        log.info(json.dumps({"event": "request", "request_id": rid, "method": request.method, "route": rpath,
                             "query": redact(str(request.url.query)), "status": response.status_code,
                             "latency_ms": round(ms, 2)}))
        return response

    # ------------------------------------------------------------------ error handling
    @app.exception_handler(MetricError)
    async def metric_error(request: Request, exc: MetricError):
        return JSONResponse(status_code=422, content={"error": "invalid_metric_request", "detail": str(exc),
                                                      "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        log.exception(json.dumps({"event": "unhandled_error", "request_id": rid, "type": type(exc).__name__}))
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})

    # ------------------------------------------------------------------ auth
    @app.post("/auth/token")
    def token(form: OAuth2PasswordRequestForm = Depends()):
        user = users.authenticate(form.username, form.password)
        if not user:
            raise HTTPException(401, "incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
        return {"access_token": create_token(auth, user), "token_type": "bearer", "role": user.role}

    @app.get("/me")
    def me(user: User = Depends(current_user)):
        return {"username": user.username, "role": user.role}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "agent_backend": agent.backend}

    # ------------------------------------------------------------------ metrics
    @app.get("/v1/metrics")
    def catalog(_: User = Depends(viewer)):
        return [{"name": d.name, "description": d.description, "params": list(d.params), "unit": d.unit}
                for d in sl.list_metrics()]

    @app.get("/v1/metrics/{name}")
    def metric(name: str, month: str | None = None, start_month: str | None = None, end_month: str | None = None,
               segment: str | None = None, measure: str | None = None, _: User = Depends(viewer)):
        kw = {k: v for k, v in dict(month=month, start_month=start_month, end_month=end_month,
                                     segment=segment, measure=measure).items() if v is not None}
        try:
            return {"metric": name, "params": kw, "value": sl.query(name, **kw)}
        except TypeError as e:
            raise MetricError(str(e)) from None

    # ------------------------------------------------------------------ agent + human in the loop
    @app.post("/v1/ask", response_model=AskOut)
    def ask(body: AskIn, user: User = Depends(viewer)):
        out = agent.ask(body.question)
        flagged = out["confidence"] < threshold and not out["refused"]
        aid = store.save_answer(user.username, out, flagged)
        rid = None
        if flagged:
            rid = store.flag(aid, "system", f"auto: confidence {out['confidence']:.2f} < {threshold}")
        telemetry.agent["refused" if out["refused"] else ("flagged" if flagged else "answered")] += 1
        return AskOut(answer_id=aid, answer=out["answer"], value=out["value"], confidence=out["confidence"],
                      flagged=flagged, review_id=rid, plan=out["plan"])

    @app.post("/v1/answers/{answer_id}/flag", status_code=201)
    def flag(answer_id: str, body: FlagIn, user: User = Depends(viewer)):
        if not store.get_answer(answer_id):
            raise HTTPException(404, "answer not found")
        existing = store.open_review_for(answer_id)
        if existing:
            return {"review_id": existing["id"], "status": "open", "duplicate": True}
        return {"review_id": store.flag(answer_id, user.username, body.reason), "status": "open", "duplicate": False}

    @app.get("/v1/reviews")
    def reviews(status: Literal["open", "resolved", "all"] = "open", _: User = Depends(admin)):
        return store.list_reviews(None if status == "all" else status)

    @app.post("/v1/reviews/{review_id}/resolve")
    def resolve(review_id: str, body: ResolveIn, user: User = Depends(admin)):
        r = store.get_review(review_id)
        if not r:
            raise HTTPException(404, "review not found")
        if r["status"] != "open":
            raise HTTPException(409, "review already resolved")
        if body.verdict == "corrected" and not body.corrected_answer:
            raise HTTPException(422, "corrected_answer is required when verdict is 'corrected'")
        store.resolve(review_id, user.username, body.verdict, body.corrected_answer, body.note)
        return store.get_review(review_id)

    @app.get("/ops/metrics", response_class=PlainTextResponse)
    def ops_metrics(_: User = Depends(admin)):
        return telemetry.prometheus()

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn copilot.api.main:app_factory --factory``."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
    return create_app()
