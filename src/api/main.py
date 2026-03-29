#!/usr/bin/env python3
"""
ForexGuard -- FastAPI Real-Time Scoring Service
=================================================
Exposes a POST /score endpoint that ingests a single raw event (portal or
trading), updates the user's rolling state via FeatureExtractor, runs the
Dense Autoencoder, and returns an anomaly verdict with explainability.

Startup:
  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Artifacts loaded at startup (via lifespan):
  models/scaler.joblib
  models/autoencoder.pth
  models/thresholds.joblib
  models/feature_names.joblib
"""

import sys
import numpy as np
import torch
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # forexguard/
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_extractor import FeatureExtractor  # noqa: E402
from src.models.train_models import DenseAutoencoder          # noqa: E402
from src.llm.risk_summarizer import RiskSummarizer, _risk_level  # noqa: E402

# ---------------------------------------------------------------------------
# Warm-up: minimum events before a user can be flagged as anomalous.
# With fewer events the deque windows are too sparse and the scaled
# features look nothing like the training distribution -> false positives.
# ---------------------------------------------------------------------------
MIN_EVENTS_FOR_SCORING = 10


# ============================================================================
# Pydantic Schemas
# ============================================================================

class EventIn(BaseModel):
    """Incoming raw event -- accepts both portal and trading fields."""
    event_id: str
    timestamp: str
    user_id: str

    # Portal fields (optional -- absent for trading events)
    event_type: Optional[str] = None
    ip_address: Optional[str] = None
    device_fingerprint: Optional[str] = None
    geo_location: Optional[str] = None
    session_duration_min: Optional[float] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    login_status: Optional[str] = None
    kyc_status: Optional[str] = None
    page_views: Optional[int] = None

    # Trading fields (optional -- absent for portal events)
    instrument: Optional[str] = None
    direction: Optional[str] = None
    lot_size: Optional[float] = None
    trade_volume_usd: Optional[float] = None
    margin_used_pct: Optional[float] = None
    pnl: Optional[float] = None
    trade_duration_sec: Optional[float] = None


class ContributorOut(BaseModel):
    """A single top-contributing feature for an anomaly."""
    feature: str
    error: float


class ScoreOut(BaseModel):
    """Response from the /score endpoint."""
    user_id: str
    is_anomalous: bool
    risk_score: float
    top_contributors: list[ContributorOut]
    risk_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated human-readable risk narrative (only for anomalous users)"
    )
    timestamp: str
    event_id: str
    events_processed: int = Field(
        description="Total events ingested for this user so far"
    )
    warming_up: bool = Field(
        default=False,
        description="True if user has fewer than MIN_EVENTS_FOR_SCORING events"
    )


# ============================================================================
# Global state -- populated during lifespan startup
# ============================================================================

class AppState:
    extractor: FeatureExtractor
    scaler: object          # StandardScaler
    model: DenseAutoencoder
    threshold: float
    feature_names: list[str]
    device: torch.device
    summarizer: RiskSummarizer
    total_scored: int = 0
    total_anomalies: int = 0


state = AppState()


# ============================================================================
# Lifespan -- load all artifacts once at startup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = PROJECT_ROOT / "models"
    print("=" * 60)
    print("  ForexGuard API -- Loading artifacts...")
    print("=" * 60)

    # 1. Feature extractor (stateful, in-memory)
    state.extractor = FeatureExtractor()
    print("  [ok] FeatureExtractor initialised")

    # 2. Scaler
    state.scaler = joblib.load(model_dir / "scaler.joblib")
    print("  [ok] StandardScaler loaded")

    # 3. Feature names
    state.feature_names = joblib.load(model_dir / "feature_names.joblib")
    print(f"  [ok] Feature names loaded ({len(state.feature_names)} features)")

    # 4. Autoencoder
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        model_dir / "autoencoder.pth",
        map_location=state.device,
        weights_only=False,
    )
    state.model = DenseAutoencoder(checkpoint["input_dim"]).to(state.device)
    state.model.load_state_dict(checkpoint["state_dict"])
    state.model.eval()
    print(f"  [ok] DenseAutoencoder loaded (device={state.device})")

    # 5. Threshold
    thresholds = joblib.load(model_dir / "thresholds.joblib")
    state.threshold = thresholds["autoencoder"]
    print(f"  [ok] Threshold loaded: {state.threshold:.4f}")
    print(f"  [ok] Warm-up: {MIN_EVENTS_FOR_SCORING} events per user before scoring")

    # 6. LLM Risk Summarizer
    state.summarizer = RiskSummarizer()
    print(f"  [ok] RiskSummarizer loaded (backend={state.summarizer.backend})")

    state.total_scored = 0
    state.total_anomalies = 0

    print("=" * 60)
    print("  Ready. Docs at http://localhost:8000/docs")
    print("=" * 60)
    yield
    print("\nForexGuard API shutting down.")


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(
    title="ForexGuard",
    description="Real-time anomaly detection for forex trader behavior",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with links to docs and health."""
    return """
    <html>
    <head><title>ForexGuard API</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 700px; margin: 60px auto; color: #1a1a2e; }
      h1 { color: #16213e; } a { color: #0f3460; } code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; }
      .stat { display: inline-block; background: #0f3460; color: white; padding: 8px 16px; border-radius: 6px; margin: 4px; }
    </style></head>
    <body>
      <h1>ForexGuard API</h1>
      <p>Real-time anomaly detection engine for forex trader behavior.</p>
      <h3>Endpoints</h3>
      <ul>
        <li><code>POST /score</code> &mdash; Score a single event with LLM risk summary (<a href="/docs#/default/score_event_score_post">try it</a>)</li>
        <li><code>GET /alert/{user_id}/summary</code> &mdash; On-demand LLM risk summary for a tracked user</li>
        <li><code>GET /health</code> &mdash; <a href="/health">Service health & stats</a></li>
        <li><code>GET /docs</code> &mdash; <a href="/docs">Interactive API docs (Swagger)</a></li>
      </ul>
      <h3>How it works</h3>
      <p>Each incoming event updates a per-user rolling state (deque windows), extracts 50 behavioral features,
         scales them, and passes through a Dense Autoencoder. High reconstruction error = anomaly.</p>
      <p>Users with fewer than <strong>10 events</strong> are in warm-up (not yet scored).</p>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "total_scored": state.total_scored,
        "total_anomalies": state.total_anomalies,
        "users_tracked": len(state.extractor.user_ids),
    }


@app.post("/score", response_model=ScoreOut)
async def score_event(event: EventIn):
    """
    Ingest a single raw event, update user state, and return anomaly verdict.

    Pipeline:
      1. event dict -> FeatureExtractor.process_event() -> 50-feature vector
      2. Drop user_id, scale with StandardScaler
      3. Forward through DenseAutoencoder -> per-feature reconstruction error
      4. MSE = anomaly score; compare to threshold
      5. If anomalous, extract top-3 contributing features

    Users with fewer than MIN_EVENTS_FOR_SCORING events are in warm-up
    and will never be flagged (warming_up=true in the response).
    """
    # -- 1. Feature extraction (updates deque state + returns features) -----
    event_dict = event.model_dump()
    features = state.extractor.process_event(event_dict, compute=True)

    user_id = features.pop("user_id")
    user_state = state.extractor._get_or_create(user_id)
    events_processed = user_state.total_events
    warming_up = events_processed < MIN_EVENTS_FOR_SCORING

    # -- 2. Build ordered feature array and scale --------------------------
    feature_vector = np.array(
        [features.get(f, 0.0) for f in state.feature_names], dtype=np.float64
    )
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = state.scaler.transform(feature_vector.reshape(1, -1))

    # -- 3. Autoencoder forward pass ----------------------------------------
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(state.device)
        reconstructed = state.model(X_tensor).cpu().numpy()

    # -- 4. Per-feature error and overall score -----------------------------
    per_feature_error = (X_scaled[0] - reconstructed[0]) ** 2
    risk_score = float(per_feature_error.mean())

    # -- 5. Anomaly verdict + explainability --------------------------------
    # Warm-up guard: never flag users with too few events
    if warming_up:
        is_anomalous = False
    else:
        is_anomalous = risk_score >= state.threshold

    top_contributors: list[ContributorOut] = []
    risk_summary: Optional[str] = None
    if is_anomalous:
        top3_idx = np.argsort(per_feature_error)[-3:][::-1]
        top_contributors = [
            ContributorOut(
                feature=state.feature_names[j],
                error=round(float(per_feature_error[j]), 4),
            )
            for j in top3_idx
        ]
        # -- 6. LLM risk summary for anomalous users -----------------------
        user_features = state.extractor.get_user_features(user_id)
        user_features.pop("user_id", None)
        risk_summary = await state.summarizer.generate(
            user_id=user_id,
            risk_score=risk_score,
            top_contributors=[c.model_dump() for c in top_contributors],
            user_features=user_features,
        )

    # -- Bookkeeping --------------------------------------------------------
    state.total_scored += 1
    if is_anomalous:
        state.total_anomalies += 1

    return ScoreOut(
        user_id=user_id,
        is_anomalous=is_anomalous,
        risk_score=round(risk_score, 6),
        top_contributors=top_contributors,
        risk_summary=risk_summary,
        timestamp=event.timestamp,
        event_id=event.event_id,
        events_processed=events_processed,
        warming_up=warming_up,
    )


# ============================================================================
# On-demand alert summary endpoint
# ============================================================================

class AlertSummaryOut(BaseModel):
    """Response from the /alert/{user_id}/summary endpoint."""
    user_id: str
    risk_score: float
    risk_level: str
    top_contributors: list[ContributorOut]
    risk_summary: str
    llm_backend: str = Field(description="Backend used: 'gemini' or 'template'")
    events_processed: int


@app.get("/alert/{user_id}/summary", response_model=AlertSummaryOut)
async def alert_summary(user_id: str):
    """
    Generate an on-demand LLM risk summary for a tracked user.

    The user must have been previously scored via POST /score (at least
    MIN_EVENTS_FOR_SCORING events). Returns a full risk narrative with
    top contributing features and recommended compliance actions.
    """
    if user_id not in state.extractor.user_ids:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found. Send events via POST /score first.")

    user_state = state.extractor._get_or_create(user_id)
    if user_state.total_events < MIN_EVENTS_FOR_SCORING:
        raise HTTPException(
            status_code=400,
            detail=f"User '{user_id}' has only {user_state.total_events} events "
                   f"(need {MIN_EVENTS_FOR_SCORING}+ for scoring)."
        )

    # Compute features and score
    user_features = state.extractor.get_user_features(user_id)
    user_features.pop("user_id", None)

    feature_vector = np.array(
        [user_features.get(f, 0.0) for f in state.feature_names], dtype=np.float64
    )
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = state.scaler.transform(feature_vector.reshape(1, -1))

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(state.device)
        reconstructed = state.model(X_tensor).cpu().numpy()

    per_feature_error = (X_scaled[0] - reconstructed[0]) ** 2
    risk_score = float(per_feature_error.mean())

    # Top contributors
    top5_idx = np.argsort(per_feature_error)[-5:][::-1]
    top_contributors = [
        ContributorOut(
            feature=state.feature_names[j],
            error=round(float(per_feature_error[j]), 4),
        )
        for j in top5_idx
    ]

    # Risk level
    risk_level = _risk_level(risk_score)

    # Generate summary
    summary = await state.summarizer.generate(
        user_id=user_id,
        risk_score=risk_score,
        top_contributors=[c.model_dump() for c in top_contributors],
        user_features=user_features,
    )

    return AlertSummaryOut(
        user_id=user_id,
        risk_score=round(risk_score, 6),
        risk_level=risk_level,
        top_contributors=top_contributors,
        risk_summary=summary,
        llm_backend=state.summarizer.backend,
        events_processed=user_state.total_events,
    )
