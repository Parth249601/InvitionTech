#!/usr/bin/env python3
"""
ForexGuard — Stateful Feature Extractor
=========================================
OOP state manager using collections.deque keyed by user_id.
Identical code path for batch (model training) and streaming (FastAPI/Kafka).

Design constraints enforced:
  1. No Pandas .rolling() — all rolling stats via bounded deques
  2. is_anomaly / anomaly_type columns are STRIPPED before processing
  3. Same process_event() call for batch and real-time

Features (48 total across 3 categories):
  Portal  (23): login patterns, IP/device deviation, session metrics,
                 deposit/withdrawal behavior, structuring signals
  Trading (18): volume/lot stats, PnL volatility, trade duration,
                 instrument concentration, direction imbalance
  Cross    (7): inter-event timing, hour entropy, dormancy, KYC-withdrawal gap

Usage:
  # ── Batch (training) ──────────────────────────────
  extractor = FeatureExtractor()
  df = extractor.extract_batch(
      "data/raw/client_portal_events.csv",
      "data/raw/trading_events.csv",
  )

  # ── Streaming (FastAPI / Kafka consumer) ──────────
  extractor = FeatureExtractor()
  features = extractor.process_event(event_dict)
"""

import math
import numpy as np
import pandas as pd
from collections import deque, Counter
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================================
# Constants
# ============================================================================

PORTAL_WINDOW = 50      # last N portal events per user
TRADING_WINDOW = 100    # last N trades per user
CROSS_WINDOW = 200      # last N events (any type) per user
UNUSUAL_HOUR_LO = 1     # 1 AM
UNUSUAL_HOUR_HI = 5     # 5 AM
SHORT_TRADE_SEC = 5.0   # latency-arbitrage threshold


# ============================================================================
# Helpers — kept minimal and branch-free where possible
# ============================================================================

def _v(val, default=0.0):
    """Return default if val is None or NaN."""
    if val is None:
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
    except TypeError:
        pass
    return val


def _parse_ts(ts) -> datetime:
    """Robust timestamp parser — handles str, Timestamp, and datetime."""
    if isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).to_pydatetime()


def _mean(dq) -> float:
    return float(np.mean(list(dq))) if dq else 0.0


def _std(dq) -> float:
    return float(np.std(list(dq), ddof=0)) if len(dq) > 1 else 0.0


def _mx(dq, default=0.0) -> float:
    return float(max(dq)) if dq else default


def _mn(dq, default=0.0) -> float:
    return float(min(dq)) if dq else default


def _entropy(values) -> float:
    """Shannon entropy (bits) of a categorical sequence."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    )


def _change_rate(dq) -> float:
    """Fraction of consecutive pairs that differ (IP-switching, device-switching)."""
    if len(dq) < 2:
        return 0.0
    items = list(dq)
    changes = sum(1 for a, b in zip(items, items[1:]) if a != b)
    return changes / (len(items) - 1)


def _inter_times(ts_deque) -> list[float]:
    """Inter-event deltas in seconds from an ordered timestamp deque."""
    if len(ts_deque) < 2:
        return []
    ts = list(ts_deque)
    return [(ts[i + 1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]


# ============================================================================
# UserState — the per-user in-memory rolling window store
# ============================================================================

class UserState:
    """
    Maintains ALL rolling state for a single user via bounded deques.
    Features are computed lazily from these deques — no redundant caches.

    This is the object that lives in the FeatureExtractor's dict and gets
    reused identically in batch replay and live streaming.
    """

    __slots__ = [
        # Portal
        "login_timestamps", "login_ips", "login_devices", "login_geos",
        "login_statuses", "session_durations", "page_views_list",
        "deposit_amounts", "deposit_timestamps",
        "withdrawal_amounts", "withdrawal_timestamps",
        "kyc_timestamps",
        # Trading
        "trade_timestamps", "trade_volumes", "lot_sizes", "pnls",
        "trade_durations", "instruments", "directions", "margins",
        # Cross-domain
        "all_event_timestamps", "activity_hours",
        # Scalar trackers (not windowed — monotonic or max-ever)
        "total_events", "current_failed_streak", "max_failed_streak",
    ]

    def __init__(self):
        # ── Portal deques ─────────────────────────────────────────────
        self.login_timestamps:    deque = deque(maxlen=PORTAL_WINDOW)
        self.login_ips:           deque = deque(maxlen=PORTAL_WINDOW)
        self.login_devices:       deque = deque(maxlen=PORTAL_WINDOW)
        self.login_geos:          deque = deque(maxlen=PORTAL_WINDOW)
        self.login_statuses:      deque = deque(maxlen=PORTAL_WINDOW)
        self.session_durations:   deque = deque(maxlen=PORTAL_WINDOW)
        self.page_views_list:     deque = deque(maxlen=PORTAL_WINDOW)
        self.deposit_amounts:     deque = deque(maxlen=PORTAL_WINDOW)
        self.deposit_timestamps:  deque = deque(maxlen=PORTAL_WINDOW)
        self.withdrawal_amounts:  deque = deque(maxlen=PORTAL_WINDOW)
        self.withdrawal_timestamps: deque = deque(maxlen=PORTAL_WINDOW)
        self.kyc_timestamps:      deque = deque(maxlen=PORTAL_WINDOW)

        # ── Trading deques ────────────────────────────────────────────
        self.trade_timestamps:    deque = deque(maxlen=TRADING_WINDOW)
        self.trade_volumes:       deque = deque(maxlen=TRADING_WINDOW)
        self.lot_sizes:           deque = deque(maxlen=TRADING_WINDOW)
        self.pnls:                deque = deque(maxlen=TRADING_WINDOW)
        self.trade_durations:     deque = deque(maxlen=TRADING_WINDOW)
        self.instruments:         deque = deque(maxlen=TRADING_WINDOW)
        self.directions:          deque = deque(maxlen=TRADING_WINDOW)
        self.margins:             deque = deque(maxlen=TRADING_WINDOW)

        # ── Cross-domain ──────────────────────────────────────────────
        self.all_event_timestamps: deque = deque(maxlen=CROSS_WINDOW)
        self.activity_hours:       deque = deque(maxlen=CROSS_WINDOW)

        # ── Scalar trackers ───────────────────────────────────────────
        self.total_events = 0
        self.current_failed_streak = 0
        self.max_failed_streak = 0

    # ────────────────────────────────────────────────────────────────────
    # State update methods (called once per event, O(1) amortised)
    # ────────────────────────────────────────────────────────────────────

    def update_portal(self, event: dict, ts: datetime) -> None:
        etype = str(_v(event.get("event_type"), ""))

        if etype == "login":
            status = str(_v(event.get("login_status"), ""))
            self.login_timestamps.append(ts)
            self.login_ips.append(str(_v(event.get("ip_address"), "")))
            self.login_devices.append(str(_v(event.get("device_fingerprint"), "")))
            self.login_geos.append(str(_v(event.get("geo_location"), "")))
            self.login_statuses.append(status)

            if status == "success":
                self.session_durations.append(float(_v(event.get("session_duration_min"), 0)))
                self.page_views_list.append(int(_v(event.get("page_views"), 0)))
                self.current_failed_streak = 0
            elif status == "failed":
                self.current_failed_streak += 1
                self.max_failed_streak = max(
                    self.max_failed_streak, self.current_failed_streak
                )

        elif etype == "deposit":
            self.deposit_amounts.append(float(_v(event.get("amount"), 0)))
            self.deposit_timestamps.append(ts)

        elif etype == "withdrawal":
            self.withdrawal_amounts.append(float(_v(event.get("amount"), 0)))
            self.withdrawal_timestamps.append(ts)

        elif etype == "kyc_change":
            self.kyc_timestamps.append(ts)

    def update_trading(self, event: dict, ts: datetime) -> None:
        self.trade_timestamps.append(ts)
        self.trade_volumes.append(float(_v(event.get("trade_volume_usd"), 0)))
        self.lot_sizes.append(float(_v(event.get("lot_size"), 0)))
        self.pnls.append(float(_v(event.get("pnl"), 0)))
        self.trade_durations.append(float(_v(event.get("trade_duration_sec"), 0)))
        self.instruments.append(str(_v(event.get("instrument"), "")))
        self.directions.append(str(_v(event.get("direction"), "")))
        self.margins.append(float(_v(event.get("margin_used_pct"), 0)))

    def update_cross_domain(self, ts: datetime) -> None:
        self.all_event_timestamps.append(ts)
        self.activity_hours.append(ts.hour)
        self.total_events += 1

    # ────────────────────────────────────────────────────────────────────
    # Feature computation (pure functions over deque state)
    # ────────────────────────────────────────────────────────────────────

    def portal_features(self) -> dict:
        n_logins = len(self.login_statuses)
        n_failed = list(self.login_statuses).count("failed")

        total_dep = sum(self.deposit_amounts) if self.deposit_amounts else 0.0
        total_wd  = sum(self.withdrawal_amounts) if self.withdrawal_amounts else 0.0
        n_dep = len(self.deposit_amounts)

        return {
            # ── Login patterns ────────────────────────────────────────
            "login_count":            n_logins,
            "failed_login_count":     n_failed,
            "failed_login_ratio":     n_failed / n_logins if n_logins else 0.0,
            "max_failed_streak":      self.max_failed_streak,

            # ── IP / Device / Geo deviation scoring ───────────────────
            "unique_ip_count":        len(set(self.login_ips)),
            "unique_device_count":    len(set(self.login_devices)),
            "unique_geo_count":       len(set(self.login_geos)),
            "ip_change_rate":         _change_rate(self.login_ips),
            "device_change_rate":     _change_rate(self.login_devices),

            # ── Unusual-hour logins (1–5 AM) ──────────────────────────
            "unusual_hour_ratio": (
                sum(1 for t in self.login_timestamps
                    if UNUSUAL_HOUR_LO <= t.hour <= UNUSUAL_HOUR_HI)
                / n_logins if n_logins else 0.0
            ),

            # ── Session-based behavioural metrics ─────────────────────
            "avg_session_duration":   _mean(self.session_durations),
            "std_session_duration":   _std(self.session_durations),
            "avg_page_views":         _mean(self.page_views_list),
            "page_view_velocity": (
                _mean(self.page_views_list)
                / max(_mean(self.session_durations), 0.1)
            ),

            # ── Deposit behaviour ─────────────────────────────────────
            "deposit_count":          n_dep,
            "total_deposit_amt":      total_dep,
            "avg_deposit_amt":        _mean(self.deposit_amounts),
            "std_deposit_amt":        _std(self.deposit_amounts),
            "small_deposit_ratio": (
                sum(1 for a in self.deposit_amounts if a < 1000)
                / n_dep if n_dep else 0.0
            ),

            # ── Withdrawal behaviour ──────────────────────────────────
            "withdrawal_count":       len(self.withdrawal_amounts),
            "total_withdrawal_amt":   total_wd,
            "max_single_withdrawal":  _mx(self.withdrawal_amounts),
            "net_deposit_flow":       total_dep - total_wd,

            # ── KYC ──────────────────────────────────────────────────
            "kyc_change_count":       len(self.kyc_timestamps),
        }

    def trading_features(self) -> dict:
        n = len(self.trade_timestamps)

        # Instrument concentration (Herfindahl-like: max share of any single instrument)
        inst_counts = Counter(self.instruments)
        concentration = max(inst_counts.values()) / n if n and inst_counts else 0.0

        # Direction imbalance |buys − sells| / total
        dc = Counter(self.directions)
        imbalance = abs(dc.get("buy", 0) - dc.get("sell", 0)) / n if n else 0.0

        # Inter-trade times
        itt = _inter_times(self.trade_timestamps)

        return {
            "trade_count":            n,
            "avg_lot_size":           _mean(self.lot_sizes),
            "std_lot_size":           _std(self.lot_sizes),
            "max_lot_size":           _mx(self.lot_sizes),
            "avg_trade_volume":       _mean(self.trade_volumes),
            "std_trade_volume":       _std(self.trade_volumes),
            "volume_spike_ratio": (
                _mx(self.trade_volumes) / max(_mean(self.trade_volumes), 1.0)
            ),
            "avg_margin_used":        _mean(self.margins),
            "avg_pnl":                _mean(self.pnls),
            "std_pnl":                _std(self.pnls),
            "pnl_win_rate": (
                sum(1 for p in self.pnls if p > 0) / n if n else 0.0
            ),
            "avg_trade_duration":     _mean(self.trade_durations),
            "min_trade_duration":     _mn(self.trade_durations),
            "short_trade_ratio": (
                sum(1 for d in self.trade_durations if d < SHORT_TRADE_SEC) / n
                if n else 0.0
            ),
            "instrument_count":       len(set(self.instruments)),
            "instrument_concentration": concentration,
            "direction_imbalance":    imbalance,
            "avg_inter_trade_time":   float(np.mean(itt)) if itt else 0.0,
        }

    def cross_domain_features(self) -> dict:
        iet = _inter_times(self.all_event_timestamps)
        dormancy_days = (max(iet) / 86400.0) if iet else 0.0

        total_dep = sum(self.deposit_amounts) if self.deposit_amounts else 0.0
        total_vol = sum(self.trade_volumes) if self.trade_volumes else 0.0

        # Min gap (hours) between ANY kyc_change and a SUBSEQUENT withdrawal
        kyc_wd_gap = self._min_kyc_to_withdrawal_hours()

        return {
            "avg_inter_event_time":   float(np.mean(iet)) if iet else 0.0,
            "std_inter_event_time":   float(np.std(iet)) if len(iet) > 1 else 0.0,
            "min_inter_event_time":   min(iet) if iet else 0.0,
            "activity_hour_entropy":  _entropy(list(self.activity_hours)),
            "dormancy_max_days":      dormancy_days,
            "total_event_count":      self.total_events,
            "deposit_to_trade_ratio": total_dep / max(total_vol, 1.0),
            "kyc_to_withdrawal_hours": kyc_wd_gap,
        }

    # ── Private helpers ───────────────────────────────────────────────

    def _min_kyc_to_withdrawal_hours(self) -> float:
        """Smallest positive gap between a KYC change and a later withdrawal.
        Returns -1.0 sentinel if no such pair exists."""
        if not self.kyc_timestamps or not self.withdrawal_timestamps:
            return -1.0
        best = float("inf")
        for kt in self.kyc_timestamps:
            for wt in self.withdrawal_timestamps:
                gap_h = (wt - kt).total_seconds() / 3600.0
                if 0 < gap_h < best:
                    best = gap_h
        return best if best != float("inf") else -1.0


# ============================================================================
# FeatureExtractor — the public API
# ============================================================================

class FeatureExtractor:
    """
    Stateful feature extractor backed by per-user deque rolling windows.

    Dual-mode operation:
      Batch     → extract_batch() loads CSVs, strips labels, replays events
      Streaming → process_event() ingests one event, returns feature vector

    The is_anomaly / anomaly_type columns are NEVER seen by this class.
    They are stripped at the boundary in extract_batch() and must never
    appear in streaming event dicts.
    """

    def __init__(self):
        self._states: dict[str, UserState] = {}

    # ── Core API ──────────────────────────────────────────────────────

    def process_event(self, event: dict, compute: bool = True) -> Optional[dict]:
        """
        Ingest a single event and optionally return the user's full feature vector.

        Args:
            event:   Dict with at minimum {user_id, event_id, timestamp, ...}
            compute: True  → return 48-feature dict (streaming / scoring path)
                     False → update state only, return None (batch replay path)
        """
        user_id = str(event.get("user_id", ""))
        if not user_id:
            return None

        state = self._get_or_create(user_id)
        ts = _parse_ts(event.get("timestamp"))

        # Route to the correct updater via event_id prefix
        eid = str(_v(event.get("event_id"), ""))
        if eid.startswith("PE_"):
            state.update_portal(event, ts)
        elif eid.startswith("TE_"):
            state.update_trading(event, ts)

        state.update_cross_domain(ts)

        if compute:
            return self.get_user_features(user_id)
        return None

    def get_user_features(self, user_id: str) -> dict:
        """Snapshot the full 48-feature vector for a user at the current moment."""
        state = self._get_or_create(user_id)
        features = {"user_id": user_id}
        features.update(state.portal_features())
        features.update(state.trading_features())
        features.update(state.cross_domain_features())
        return features

    # ── Batch pipeline ────────────────────────────────────────────────

    def extract_batch(
        self,
        portal_path: str,
        trading_path: str,
    ) -> pd.DataFrame:
        """
        Full batch extraction pipeline:
          1. Load CSVs
          2. Strip is_anomaly / anomaly_type (unsupervised discipline)
          3. Merge both sources, sort chronologically
          4. Replay every event through process_event(compute=False)
          5. Return one feature row per user

        This is the entry point for training-data preparation.
        """
        print("[FeatureExtractor] Loading data...")
        portal = pd.read_csv(portal_path)
        trading = pd.read_csv(trading_path)

        # ── STRICT UNSUPERVISED DISCIPLINE ────────────────────────────
        for col in ("is_anomaly", "anomaly_type"):
            portal.drop(columns=[col], inplace=True, errors="ignore")
            trading.drop(columns=[col], inplace=True, errors="ignore")
        print(f"  Portal events : {len(portal):,}")
        print(f"  Trading events: {len(trading):,}")
        print("  Labels stripped: is_anomaly, anomaly_type")

        # ── Merge & sort ──────────────────────────────────────────────
        print("[FeatureExtractor] Merging & sorting events chronologically...")
        all_events = pd.concat([portal, trading], ignore_index=True)
        all_events.sort_values("timestamp", inplace=True)
        records = all_events.to_dict("records")
        total = len(records)

        # ── Replay (state-update only — no feature computation) ───────
        print(f"[FeatureExtractor] Replaying {total:,} events...")
        for i, event in enumerate(records):
            self.process_event(event, compute=False)
            if (i + 1) % 10_000 == 0:
                print(f"  {i + 1:>7,} / {total:,}")

        # ── Extract final snapshot per user ───────────────────────────
        n_users = len(self._states)
        print(f"[FeatureExtractor] Computing features for {n_users} users...")
        rows = [self.get_user_features(uid) for uid in sorted(self._states)]
        df = pd.DataFrame(rows)
        n_feats = df.shape[1] - 1  # minus user_id
        print(f"  Feature matrix: {n_users} users x {n_feats} features")
        return df

    # ── Utilities ─────────────────────────────────────────────────────

    def reset(self):
        """Clear all user states (e.g. between experiments)."""
        self._states.clear()

    @property
    def user_ids(self) -> list[str]:
        return sorted(self._states.keys())

    def _get_or_create(self, user_id: str) -> UserState:
        if user_id not in self._states:
            self._states[user_id] = UserState()
        return self._states[user_id]


# ============================================================================
# CLI entry point — run batch extraction
# ============================================================================

def main():
    base = Path(__file__).resolve().parent.parent.parent  # forexguard/
    raw_dir = base / "data" / "raw"
    out_dir = base / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract features ──────────────────────────────────────────────
    extractor = FeatureExtractor()
    features_df = extractor.extract_batch(
        str(raw_dir / "client_portal_events.csv"),
        str(raw_dir / "trading_events.csv"),
    )

    # ── Save training set (NO labels) ─────────────────────────────────
    features_df.to_csv(out_dir / "user_features.csv", index=False)

    # ── Save evaluation set (labels merged — for Phase 3 metrics only)
    gt = pd.read_csv(raw_dir / "ground_truth.csv")
    merged = features_df.merge(
        gt[["user_id", "is_anomaly", "anomaly_types"]], on="user_id", how="left"
    )
    merged.to_csv(out_dir / "user_features_with_labels.csv", index=False)

    print(f"\nSaved:")
    print(f"  {out_dir / 'user_features.csv'}  (training — no labels)")
    print(f"  {out_dir / 'user_features_with_labels.csv'}  (evaluation only)")

    # ── Quick sanity stats ────────────────────────────────────────────
    numeric = features_df.select_dtypes(include=[np.number])
    print(f"\nFeature statistics ({numeric.shape[1]} numeric features):")
    print(numeric.describe().round(3).to_string())


if __name__ == "__main__":
    main()
