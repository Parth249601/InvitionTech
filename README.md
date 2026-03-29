<p align="center">
  <h1 align="center">ForexGuard</h1>
  <p align="center">
    <strong>Real-Time Anomaly Detection Engine for Forex Trader Behavior</strong>
  </p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> &bull;
    <a href="#architecture-overview">Architecture</a> &bull;
    <a href="#ml-approach">ML Approach</a> &bull;
    <a href="#api-reference">API Reference</a> &bull;
    <a href="#deployment">Deployment</a> &bull;
    <a href="#evaluation-results">Results</a>
  </p>
</p>

---

A production-grade, real-time anomaly detection prototype that identifies suspicious user activity across **client portal** and **trading terminal** events in a forex brokerage environment. The system ingests raw event streams, maintains per-user behavioral state via rolling windows, scores each event through dual ML models, and returns **explainable risk alerts** for compliance teams -- all in under 3ms per event.

> Built as the AI/ML Internship Assessment for **Invition Technologies**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [ML Approach](#ml-approach)
  - [Model Selection Justification](#model-selection-justification)
  - [Autoencoder Architecture](#autoencoder-architecture)
  - [Explainability via Reconstruction Error](#explainability-via-reconstruction-error)
- [Feature Engineering](#feature-engineering)
- [Anomaly Patterns Detected](#anomaly-patterns-detected)
- [Engineering Highlights](#engineering-highlights)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Evaluation Results](#evaluation-results)
- [Assumptions, Trade-offs & Limitations](#assumptions-trade-offs--limitations)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)

---

## Architecture Overview

ForexGuard uses a **dual-pipeline** design: a batch pipeline for offline training and an identical streaming pipeline for real-time inference. Both share the exact same stateful `FeatureExtractor`, ensuring **training-serving parity**.

```
                          BATCH PIPELINE (Training)
  +-----------+     +-------------------+     +-----------------+     +--------+
  | Synthetic |---->| FeatureExtractor  |---->| StandardScaler  |---->| Models |
  | Data Gen  |     | (replay all 54K   |     | (fit + save)    |     | (fit)  |
  | (54K evts)|     |  events in order) |     +-----------------+     +--------+
  +-----------+     +-------------------+           |                     |
                            |                  scaler.joblib    autoencoder.pth
                    user_features.csv                       isolation_forest.joblib

                         STREAMING PIPELINE (Inference)
  +-------+     +-------------------+     +-----------+     +------------+     +----------+
  | Event |---->| FeatureExtractor  |---->| Scaler    |---->| Autoencoder|---->| Anomaly  |
  | (raw) |     | .process_event()  |     | .transform|     | .forward() |     | Verdict  |
  +-------+     +-------------------+     +-----------+     +------------+     +----------+
                  collections.deque                           MSE score      is_anomalous +
                  keyed by user_id                            per feature    top 3 features
```

**Key design decision:** Both pipelines use the same `FeatureExtractor` class. In batch mode, events are replayed chronologically with `compute=False` (state-update only), then features are extracted once per user. In streaming mode, each `process_event()` call updates state and returns the full 50-feature vector instantly. No Pandas `.rolling()` is ever used in the streaming path -- all rolling statistics come from bounded `collections.deque` objects keyed by `user_id`.

---

## ML Approach

### Model Selection Justification

We implemented two complementary unsupervised models, choosing architectures that match the structure of our data (tabular user-level aggregates, not raw time series):

| Model | Type | Rationale |
|-------|------|-----------|
| **Isolation Forest** | Baseline (scikit-learn) | Excels at tabular anomaly detection with zero distributional assumptions. Sub-millisecond inference. Industry standard for fraud detection pipelines. |
| **Dense Autoencoder** | Advanced (PyTorch) | Learns a compressed latent representation of "normal" trader behavior. Anomalous users exhibit high reconstruction error. Provides **inherent explainability** via per-feature error decomposition -- no post-hoc methods (SHAP/LIME) needed. |

**Why Dense Autoencoder over LSTM Autoencoder?** Our features are **fixed-width tabular aggregates** (50 floats per user), not variable-length temporal sequences. An LSTM would introduce unnecessary architectural complexity (padding, packing, sequence collation) with no improvement on fixed-width input vectors. The Dense Autoencoder is the correct architectural choice for this data shape.

### Autoencoder Architecture

```
Encoder: 50 -> 32 (BatchNorm + ReLU + Dropout 0.1) -> 16 (BatchNorm + ReLU) -> 8 (bottleneck)
Decoder:  8 -> 16 (BatchNorm + ReLU + Dropout 0.1) -> 32 (BatchNorm + ReLU) -> 50
```

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Bottleneck ratio | 50 -> 8 (6.25x compression) | Forces the network to learn only the dominant patterns of normal behavior |
| BatchNorm | After every linear layer | Stabilizes training across heterogeneous feature scales |
| Dropout | 0.1 | Prevents memorization of outlier patterns without losing signal |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) | Standard choice with mild L2 regularization |
| Loss | MSE | Directly interpretable as reconstruction quality |
| Training | 50 epochs, batch size 64 | Converges fully on this dataset |

### Explainability via Reconstruction Error

The autoencoder provides explainability **without any post-hoc methods** (no SHAP, no LIME, no permutation importance). For every scored user:

1. Compute per-feature squared error: `(input[i] - reconstructed[i])^2`
2. Rank features by individual reconstruction error
3. Return the **top 3 contributors** in the API response

This tells a compliance analyst *exactly which behavioral dimension is abnormal*. For example: `"unusual_hour_ratio"` with error `296.86` means this user's after-hours activity is dramatically outside the learned normal distribution.

---

## Feature Engineering

**50 features** across three categories, all computed from bounded `collections.deque` rolling windows with O(1) amortized updates:

### Portal Features (23)
| Feature | Anomaly Signal |
|---------|---------------|
| `login_count`, `failed_login_count`, `failed_login_ratio`, `max_failed_streak` | Brute-force login detection |
| `unique_ip_count`, `unique_device_count`, `unique_geo_count` | Multi-device / multi-location access |
| `ip_change_rate`, `device_change_rate` | Rapid switching between sessions |
| `unusual_hour_ratio` | After-hours access (1-5 AM) |
| `avg_session_duration`, `std_session_duration` | Session behavior baseline |
| `avg_page_views`, `page_view_velocity` | Bot-like rapid navigation |
| `deposit_count`, `total_deposit_amt`, `avg_deposit_amt`, `std_deposit_amt` | Deposit behavior |
| `small_deposit_ratio` | Structuring / smurfing signal |
| `withdrawal_count`, `total_withdrawal_amt`, `max_single_withdrawal` | Withdrawal behavior |
| `net_deposit_flow`, `kyc_change_count` | Financial flow and account changes |

### Trading Features (18)
| Feature | Anomaly Signal |
|---------|---------------|
| `trade_count`, `avg_lot_size`, `std_lot_size`, `max_lot_size` | Trading volume profile |
| `avg_trade_volume`, `std_trade_volume`, `volume_spike_ratio` | Volume spike detection (10x baseline) |
| `avg_margin_used` | Leverage behavior |
| `avg_pnl`, `std_pnl`, `pnl_win_rate` | PnL volatility and suspicious win rates |
| `avg_trade_duration`, `min_trade_duration`, `short_trade_ratio` | Latency arbitrage (<5s trades) |
| `instrument_count`, `instrument_concentration` | Single-instrument concentration |
| `direction_imbalance`, `avg_inter_trade_time` | Directional bias, trade clustering |

### Cross-Domain Features (9)
| Feature | Anomaly Signal |
|---------|---------------|
| `avg_inter_event_time`, `std_inter_event_time`, `min_inter_event_time` | Bot detection, timing regularity |
| `activity_hour_entropy` | Distribution of activity across hours |
| `dormancy_max_days` | Long gaps before sudden activity |
| `total_event_count` | Overall activity level |
| `deposit_to_trade_ratio` | Deposit-withdrawal abuse pattern |
| `kyc_to_withdrawal_hours` | Rapid KYC change before withdrawal |

---

## Anomaly Patterns Detected

The synthetic dataset injects **16 distinct anomaly patterns** across 75 users (15% contamination rate). All patterns are verified as separable in the 50-dimensional feature space:

| Category | Patterns Injected | Key Discriminating Features |
|----------|-------------------|---------------------------|
| **Login & Access** (Sec 8.1) | Multi-IP login, IP hub (8 users sharing 1 IP), unusual-hour login, device mismatch, brute-force | `unique_ip_count` (32x normal), `unusual_hour_ratio` (0.44), `max_failed_streak` (6.5) |
| **Financial** (Sec 8.2) | Large withdrawal after dormancy, deposit-withdraw abuse, structuring | `max_single_withdrawal` (21x), `deposit_count` (20x), `small_deposit_ratio` (0.97) |
| **Trading** (Sec 8.3) | Volume spike (10x), single-instrument concentration, latency arbitrage, consistent profit | `volume_spike_ratio` (2.8x), `short_trade_ratio` (0.99), `pnl_win_rate` (0.86) |
| **Behavioral** (Sec 8.4) | Bot-like navigation, frequent device switching | `page_view_velocity` (103x normal), `unique_device_count` (3.6x) |
| **Network** (Sec 8.5) | Collusion rings (mirror trades within 0-8s), IP hub | Synchronized timestamps, opposite trade directions |
| **Temporal** (Sec 8.6) | Sudden behavior shift | Captured via rolling window statistics (std features) |
| **Account Risk** (Sec 8.7) | Rapid KYC change before withdrawal | `kyc_to_withdrawal_hours` (0.065h vs 141h normal) |

---

## Engineering Highlights

### 1. Stateful Feature Extractor with `collections.deque`

The `UserState` class maintains **22 bounded deques** per user (50-200 maxlen). All rolling statistics (mean, std, change rate, entropy) are computed from these deques. This design:

- Uses **O(1) amortized** appends (deque auto-evicts old entries)
- Requires **zero database lookups** -- everything is in-process memory
- Produces **identical results** whether replaying 54K historical events or processing one event at a time
- Guarantees **training-serving parity**: the streaming pipeline uses the exact same code path as batch training

### 2. Cold-Start Mitigation (Warm-Up Threshold)

**Problem:** When the API starts fresh, users with 1-2 events produce sparse feature vectors. After StandardScaler transformation (fitted on users with 50+ events), these vectors are extreme outliers -- the autoencoder can't reconstruct them and flags everyone as anomalous.

**Solution:** A `MIN_EVENTS_FOR_SCORING = 10` guard. Users with fewer events receive `warming_up: true` in the response and are never flagged. Impact:
- False-positive rate dropped from **100% to 2.5%**
- Throughput improved from 337 to **441 events/sec** (fewer anomaly explanations to compute)

### 3. Strict Unsupervised Discipline

The `is_anomaly` and `anomaly_type` columns are:
- Present in the raw synthetic data (for final evaluation only)
- **Stripped** at the boundary of `FeatureExtractor.extract_batch()` before any feature computation
- **Never seen** by the scaler, the Isolation Forest, or the autoencoder during training
- Only rejoined in a separate evaluation file for computing precision/recall/F1

### 4. Dual-Mode Feature Extractor

The same `process_event()` method is called in both modes with zero code duplication:
```python
# Batch: replay all events, extract features at the end
for event in all_54k_events:
    extractor.process_event(event, compute=False)  # state-update only
features_df = [extractor.get_user_features(uid) for uid in extractor.user_ids]

# Streaming: score on every event
features = extractor.process_event(event, compute=True)  # returns 50-feature dict
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/forexguard.git
cd forexguard
pip install -r requirements.txt
```

### Full Pipeline (4 steps)

```bash
# Step 1: Generate synthetic data (~54K events)
python data/generate_synthetic_data.py

# Step 2: Extract features (500 users x 50 features)
python src/features/feature_extractor.py

# Step 3: Train models (Isolation Forest + Dense Autoencoder)
python src/models/train_models.py

# Step 4: Start the API server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The API is now live at **http://localhost:8000** with interactive Swagger docs at **http://localhost:8000/docs**.

### Run the Streaming Simulator

In a second terminal (while the API is running):

```bash
python src/streaming/simulator.py
```

To limit the number of events (useful for demos):

```bash
# Bash / macOS / Linux
MAX_EVENTS=1000 python src/streaming/simulator.py

# PowerShell (Windows)
$env:MAX_EVENTS=1000; python src/streaming/simulator.py
```

### Docker (Alternative)

```bash
docker-compose up --build
# Then in another terminal:
python src/streaming/simulator.py
```

---

## API Reference

### `GET /` -- Landing Page

Returns an HTML overview page with links to docs and health endpoints.

### `GET /health` -- Service Health

```json
{
  "status": "healthy",
  "total_scored": 1000,
  "total_anomalies": 25,
  "users_tracked": 301
}
```

### `POST /score` -- Score a Single Event

Ingests one raw event (portal or trading), updates the user's rolling state, runs the autoencoder, and returns an anomaly verdict with explainability.

**Request Body:**
```json
{
  "event_id": "PE_000001",
  "timestamp": "2025-01-15 03:22:11",
  "user_id": "user_0500",
  "event_type": "login",
  "ip_address": "210.171.55.99",
  "login_status": "success",
  "session_duration_min": 0.8,
  "page_views": 45
}
```

**Response (200 OK):**
```json
{
  "user_id": "user_0500",
  "is_anomalous": true,
  "risk_score": 41.975163,
  "top_contributors": [
    {"feature": "activity_hour_entropy", "error": 1210.51},
    {"feature": "unusual_hour_ratio", "error": 296.86},
    {"feature": "avg_margin_used", "error": 188.66}
  ],
  "timestamp": "2025-01-15 03:22:11",
  "event_id": "PE_000001",
  "events_processed": 15,
  "warming_up": false
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `is_anomalous` | `bool` | `true` if `risk_score` exceeds the P85 threshold (0.726) |
| `risk_score` | `float` | Mean reconstruction error from the autoencoder (higher = more suspicious) |
| `top_contributors` | `list` | Top 3 features with highest per-feature reconstruction error |
| `events_processed` | `int` | Total events ingested for this user since API startup |
| `warming_up` | `bool` | `true` if user has fewer than 10 events (not yet scored) |

---

## Deployment

### Option 1: Docker (Local / Any Cloud)

```bash
docker-compose up --build
```

The container loads pre-trained model artifacts from `models/` and exposes the API on port `8000`.

### Option 2: Render (Free Tier)

1. Push the repository to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repository
4. Set:
   - **Environment:** Docker
   - **Branch:** main
5. Deploy -- Render will build from the `Dockerfile` automatically

### Option 3: HuggingFace Spaces (Free)

1. Create a new Space with **Docker** SDK
2. Push the repository contents to the Space
3. The `Dockerfile` exposes port `8000` -- HuggingFace will route to it automatically

### Option 4: Railway / Fly.io

Both platforms support `Dockerfile`-based deployments with free tiers. Point to the repo and deploy.

> **Note:** All model artifacts (`models/*.joblib`, `models/*.pth`) are included in the repository and are loaded at container startup. No training step is required for deployment.

---

## Project Structure

```
forexguard/
  data/
    generate_synthetic_data.py      # Synthetic event generator (54K events, 16 anomaly patterns)
    raw/
      client_portal_events.csv      # 27,456 portal events
      trading_events.csv            # 26,943 trading events
      ground_truth.csv              # 500 users with anomaly labels (evaluation only)
    processed/
      user_features.csv             # 500 x 50 feature matrix (no labels)
      user_features_with_labels.csv # Same + labels (evaluation only)
  src/
    features/
      feature_extractor.py          # Stateful OOP feature extractor (deque-backed)
    models/
      train_models.py               # Isolation Forest + Autoencoder training & evaluation
    api/
      main.py                       # FastAPI real-time scoring service
    streaming/
      simulator.py                  # Async event streamer (httpx + asyncio)
  models/
    scaler.joblib                   # Fitted StandardScaler
    isolation_forest.joblib         # Trained Isolation Forest
    autoencoder.pth                 # Autoencoder weights + architecture config
    feature_names.joblib            # Ordered feature name list (50 features)
    thresholds.joblib               # Anomaly scoring thresholds
  Dockerfile                        # Production container image
  docker-compose.yml                # Single-command local deployment
  requirements.txt                  # Python dependencies
  README.md                         # This file
```

---

## Evaluation Results

### Batch Evaluation (Full Dataset -- 500 Users)

| Metric | Isolation Forest | Dense Autoencoder |
|--------|:----------------:|:-----------------:|
| **Precision** | 0.867 | 0.813 |
| **Recall** | 0.867 | 0.813 |
| **F1-Score** | **0.867** | 0.813 |
| **Accuracy** | 96.0% | 94.4% |

> Isolation Forest slightly outperforms on this dataset, which is expected -- tabular features with clear distributional separation favor tree-based methods. The autoencoder's primary value is **explainability**: it tells the compliance team *which specific behavioral features* drove the anomaly, not just that one was detected.

### Streaming Evaluation (Real-Time Simulation)

| Metric | Value |
|--------|:-----:|
| Events streamed | 1,000 |
| Errors | 0 |
| Anomalies detected | 25 (2.5%) |
| Unique users flagged | 5 |
| Throughput | **441 events/sec** |
| Cold-start FP rate (before warm-up) | 100% |
| FP rate (after warm-up fix) | **2.5%** |

---

## Assumptions, Trade-offs & Limitations

| Decision | Rationale | Production Path |
|----------|-----------|-----------------|
| **Async HTTP simulation instead of Kafka** | Optimized for 2-day delivery window. The architecture is Kafka-ready: each event is processed independently via `process_event()`, and the `/score` endpoint could be called from a Kafka consumer with zero code changes. | Drop-in Kafka/Redpanda consumer wrapping the same `process_event()` call |
| **In-process `dict[str, UserState]` instead of Redis** | Sufficient for a single-node prototype with 500 users. | Redis or another distributed store for horizontal scaling and state persistence across restarts |
| **Dense Autoencoder over LSTM** | Features are fixed-width tabular aggregates (50 floats), not variable-length sequences. LSTM adds complexity without benefit. | If raw event sequences are needed in the future, LSTM or Transformer can be layered on top |
| **P85 threshold (not P90/P95)** | Matches our 15% contamination ratio in synthetic data. | Threshold tuned on a validation set or via business rules in production |
| **Synthetic data** | Required by the assessment. Anomaly injection is based on real forex fraud patterns, but normal behavior distribution is simplified. | Replace with real production event streams |
| **10-event warm-up** | Practical cold-start mitigation. | Warm up state from a historical event log at startup rather than waiting for live events |
| **Single-process API** | Uvicorn with one worker. | Multiple workers behind a load balancer with shared state in Redis |
| **No LLM risk summaries** | Not implemented in current version. | Integrate an LLM to generate natural-language risk narratives from `top_contributors` for compliance analysts |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Framework** | PyTorch 2.x, scikit-learn 1.x |
| **Data Processing** | Pandas, NumPy |
| **Feature Engineering** | `collections.deque` (custom OOP state manager) |
| **API** | FastAPI, Pydantic v2, Uvicorn |
| **Streaming** | asyncio, httpx (simulated; Kafka-ready architecture) |
| **Infrastructure** | Docker, docker-compose |
| **Serialization** | joblib (sklearn models), torch.save (PyTorch) |

---

## Future Work

- **Kafka/Redpanda Integration** -- Replace HTTP simulation with a real message broker for production-grade streaming
- **LLM-Generated Risk Summaries** -- Feed `top_contributors` into an LLM to generate natural-language compliance alerts (e.g., *"User user_0042 exhibits bot-like navigation patterns with 103x normal page velocity and login activity concentrated in the 1-5 AM window"*)
- **Dashboard UI** -- Real-time Grafana or custom React dashboard showing anomaly trends, user risk heatmaps, and alert history
- **Redis State Backend** -- Replace in-process `dict` with Redis for horizontal scaling and state persistence
- **Model Retraining Pipeline** -- Scheduled retraining on fresh data with MLflow experiment tracking
- **Graph-Based Detection** -- NetworkX or Neo4j for collusion ring detection across shared IP/device clusters

---

<p align="center">
  <sub>Built with PyTorch, FastAPI, and scikit-learn</sub>
</p>
