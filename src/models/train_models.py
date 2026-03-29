#!/usr/bin/env python3
"""
ForexGuard — Model Training Pipeline
======================================
Trains two anomaly detection models on the unlabeled user feature matrix,
then evaluates both against held-out ground truth labels.

Models:
  1. Isolation Forest  (sklearn baseline)   -> saved via joblib
  2. Dense Autoencoder (PyTorch advanced)   -> saved as .pth weights

Artifacts saved to models/ :
  scaler.joblib              StandardScaler fitted on training data
  isolation_forest.joblib    Trained Isolation Forest
  autoencoder.pth            Autoencoder weights
  feature_names.joblib       Ordered feature name list (for explainability)

Usage:
  python src/models/train_models.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_recall_fscore_support
from pathlib import Path
import joblib

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Isolation Forest
IF_CONTAMINATION = 0.15   # matches our known anomaly ratio
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = "auto"

# Autoencoder
AE_EPOCHS = 50
AE_BATCH_SIZE = 64
AE_LR = 1e-3
AE_WEIGHT_DECAY = 1e-5

# Evaluation threshold: top N% of anomaly scores flagged as anomalous
SCORE_PERCENTILE = 85


# ============================================================================
# Autoencoder Architecture
# ============================================================================

class DenseAutoencoder(nn.Module):
    """
    Symmetric dense autoencoder for tabular anomaly detection.

    Architecture (for 50 input features):
      Encoder: 50 -> 32 -> 16 -> 8  (bottleneck)
      Decoder: 8  -> 16 -> 32 -> 50

    Normal users reconstruct well (low MSE).
    Anomalous users reconstruct poorly (high MSE) -> anomaly score.

    Per-feature reconstruction error gives explainability:
    the features with highest |input - output|² are the top risk contributors.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================================
# Preprocessing
# ============================================================================

def load_and_preprocess(features_path: str, model_dir: Path):
    """
    Load user_features.csv -> drop user_id -> impute -> scale -> save scaler.
    Returns (scaled numpy array, feature names list, user_ids Series).
    """
    df = pd.read_csv(features_path)
    user_ids = df["user_id"]
    feature_names = [c for c in df.columns if c != "user_id"]
    X = df[feature_names].copy()

    # Impute: fill NaN with 0, replace inf with column max
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Fit and save scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(feature_names, model_dir / "feature_names.joblib")
    print(f"  Scaler saved          -> {model_dir / 'scaler.joblib'}")
    print(f"  Feature names saved   -> {model_dir / 'feature_names.joblib'}")

    return X_scaled, feature_names, user_ids


# ============================================================================
# Isolation Forest
# ============================================================================

def train_isolation_forest(X_scaled: np.ndarray, model_dir: Path) -> IsolationForest:
    """Train and persist an Isolation Forest."""
    print(f"\n{'='*60}")
    print("  ISOLATION FOREST (Baseline)")
    print(f"{'='*60}")
    print(f"  n_estimators  : {IF_N_ESTIMATORS}")
    print(f"  contamination : {IF_CONTAMINATION}")

    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        max_samples=IF_MAX_SAMPLES,
        random_state=SEED,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    joblib.dump(iso, model_dir / "isolation_forest.joblib")
    print(f"  Model saved -> {model_dir / 'isolation_forest.joblib'}")
    return iso


# ============================================================================
# Autoencoder Training
# ============================================================================

def train_autoencoder(X_scaled: np.ndarray, model_dir: Path) -> DenseAutoencoder:
    """Train the dense autoencoder and persist weights."""
    print(f"\n{'='*60}")
    print("  DENSE AUTOENCODER (Advanced)")
    print(f"{'='*60}")

    input_dim = X_scaled.shape[1]
    print(f"  Input dim     : {input_dim}")
    print(f"  Architecture  : {input_dim}->32->16->8->16->32->{input_dim}")
    print(f"  Epochs        : {AE_EPOCHS}")
    print(f"  Batch size    : {AE_BATCH_SIZE}")
    print(f"  Learning rate : {AE_LR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device        : {device}")

    # Prepare data
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    dataset = TensorDataset(X_tensor, X_tensor)  # input == target (autoencoder)
    loader = DataLoader(dataset, batch_size=AE_BATCH_SIZE, shuffle=True)

    # Init model
    model = DenseAutoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\n  {'Epoch':>5}  {'Loss':>10}")
    print(f"  {'-'*5}  {'-'*10}")

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_x, batch_target in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>5}  {avg_loss:>10.6f}")

    # Save weights
    torch.save(model.state_dict(), model_dir / "autoencoder.pth")
    # Also save the input_dim so we can reconstruct the architecture at load time
    torch.save({"input_dim": input_dim, "state_dict": model.state_dict()},
               model_dir / "autoencoder.pth")
    print(f"\n  Model saved -> {model_dir / 'autoencoder.pth'}")
    return model


# ============================================================================
# Scoring & Explainability
# ============================================================================

def score_isolation_forest(iso: IsolationForest, X_scaled: np.ndarray) -> np.ndarray:
    """
    Return anomaly scores from Isolation Forest.
    sklearn returns negative scores (more negative = more anomalous).
    We negate so higher = more anomalous.
    """
    raw_scores = iso.decision_function(X_scaled)
    return -raw_scores  # flip: higher = more anomalous


def score_autoencoder(
    model: DenseAutoencoder,
    X_scaled: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[list[tuple[str, float]]]]:
    """
    Compute per-user anomaly scores and per-feature explainability.

    Returns:
        scores: (n_users,) MSE per user (higher = more anomalous)
        explanations: list of [(feature_name, error), ...] top-3 per user
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        reconstructed = model(X_tensor).cpu().numpy()

    # Per-feature squared error -> (n_users, n_features)
    per_feature_error = (X_scaled - reconstructed) ** 2

    # Overall MSE per user -> anomaly score
    scores = per_feature_error.mean(axis=1)

    # Top-3 contributing features per user (explainability)
    explanations = []
    for i in range(len(X_scaled)):
        user_errors = per_feature_error[i]
        top3_idx = np.argsort(user_errors)[-3:][::-1]
        top3 = [(feature_names[j], float(user_errors[j])) for j in top3_idx]
        explanations.append(top3)

    return scores, explanations


def generate_alert(user_id: str, score: float, top_features: list[tuple[str, float]]) -> str:
    """Generate a human-readable risk alert for a flagged user."""
    lines = [f"ALERT: User {user_id} — Risk Score: {score:.4f}"]
    lines.append("  Top contributing risk factors:")
    for fname, ferr in top_features:
        lines.append(f"    • {fname}: error={ferr:.4f}")
    return "\n".join(lines)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model_name: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    percentile: float = SCORE_PERCENTILE,
) -> float:
    """
    Evaluate anomaly scores against ground truth using a percentile threshold.
    Prints a classification report and returns the threshold used.
    """
    threshold = np.percentile(scores, percentile)
    y_pred = (scores >= threshold).astype(int)

    print(f"\n  {model_name} — Threshold: {threshold:.4f} (P{percentile})")
    print(f"  Flagged {y_pred.sum()} / {len(y_pred)} users as anomalous")
    print()

    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Anomalous"],
        digits=3,
        zero_division=0,
    )
    print(report)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0,
    )
    print(f"  -> Anomaly class:  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
    return threshold


# ============================================================================
# Main
# ============================================================================

def main():
    base = Path(__file__).resolve().parent.parent.parent  # forexguard/
    data_dir = base / "data" / "processed"
    model_dir = base / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Preprocessing ──────────────────────────────────────────────
    print(f"{'='*60}")
    print("  PREPROCESSING")
    print(f"{'='*60}")
    X_scaled, feature_names, user_ids = load_and_preprocess(
        str(data_dir / "user_features.csv"), model_dir
    )
    print(f"  Feature matrix: {X_scaled.shape[0]} users × {X_scaled.shape[1]} features")

    # ── 2. Train Isolation Forest ─────────────────────────────────────
    iso = train_isolation_forest(X_scaled, model_dir)

    # ── 3. Train Autoencoder ──────────────────────────────────────────
    ae = train_autoencoder(X_scaled, model_dir)

    # ── 4. Score both models ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SCORING & EVALUATION")
    print(f"{'='*60}")

    # Load ground truth (evaluation only — never used in training)
    gt = pd.read_csv(data_dir / "user_features_with_labels.csv")
    label_map = gt.set_index("user_id")["is_anomaly"].to_dict()
    y_true = np.array([label_map.get(uid, 0) for uid in user_ids])
    print(f"  Ground truth: {y_true.sum()} anomalous / {len(y_true)} total")

    # Isolation Forest scores
    if_scores = score_isolation_forest(iso, X_scaled)

    # Autoencoder scores + explainability
    ae_scores, ae_explanations = score_autoencoder(ae, X_scaled, feature_names)

    # ── 5. Evaluate ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ISOLATION FOREST EVALUATION")
    print(f"{'='*60}")
    if_threshold = evaluate("Isolation Forest", if_scores, y_true)

    print(f"\n{'='*60}")
    print("  AUTOENCODER EVALUATION")
    print(f"{'='*60}")
    ae_threshold = evaluate("Autoencoder", ae_scores, y_true)

    # Save thresholds for inference
    thresholds = {
        "isolation_forest": float(if_threshold),
        "autoencoder": float(ae_threshold),
        "percentile": SCORE_PERCENTILE,
    }
    joblib.dump(thresholds, model_dir / "thresholds.joblib")
    print(f"\n  Thresholds saved -> {model_dir / 'thresholds.joblib'}")

    # ── 6. Sample alerts (top 5 riskiest users from autoencoder) ──────
    print(f"\n{'='*60}")
    print("  SAMPLE RISK ALERTS (Top 5 — Autoencoder)")
    print(f"{'='*60}\n")

    top5_idx = np.argsort(ae_scores)[-5:][::-1]
    for idx in top5_idx:
        uid = user_ids.iloc[idx]
        alert = generate_alert(uid, ae_scores[idx], ae_explanations[idx])
        label = "ACTUAL ANOMALY" if y_true[idx] else "normal"
        print(f"{alert}")
        print(f"  Ground truth: {label}")
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("  ARTIFACTS SAVED")
    print(f"{'='*60}")
    for f in sorted(model_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
