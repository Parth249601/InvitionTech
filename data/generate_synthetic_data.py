#!/usr/bin/env python3
"""
ForexGuard — Synthetic Data Generator
======================================
Generates ~50,000 realistic forex brokerage events with heavily injected
anomaly patterns covering all categories from the assessment specification.

Output:
  data/raw/client_portal_events.csv  (~25K events)
  data/raw/trading_events.csv        (~25K events)
  data/raw/ground_truth.csv          (user-level labels — evaluation ONLY)

Anomaly Coverage (Section 8):
  8.1  Login/Access    : multi_ip_login, ip_hub, unusual_time_login,
                         device_mismatch, brute_force_login
  8.2  Financial       : large_withdrawal_dormant, deposit_withdraw_abuse,
                         structuring
  8.3  Trading         : volume_spike, single_instrument, latency_arbitrage,
                         consistent_profit
  8.4  Behavioral      : bot_behavior, device_switching
  8.5  Graph/Network   : ip_hub (shared IP/device), collusion_ring (mirror trades)
  8.6  Temporal        : volume_spike (sudden shift), news_aligned_trading
  8.7  Account Risk    : rapid_kyc_withdrawal, brute_force_login

Usage:
  python data/generate_synthetic_data.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
NUM_USERS = 500
ANOMALY_RATIO = 0.15  # 15% anomalous users → 75 users
START = datetime(2025, 1, 1)
END = datetime(2025, 3, 31)  # 90-day observation window
WINDOW_DAYS = (END - START).days

INSTRUMENTS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CHF", "EUR/GBP", "NZD/USD", "USD/CAD",
]

REGION_IP_PREFIX = {
    "US_East": "72.34",
    "US_West": "64.18",
    "UK": "81.2",
    "Germany": "85.214",
    "Singapore": "103.21",
    "Japan": "210.171",
    "UAE": "94.56",
    "Australia": "103.4",
}
REGIONS = list(REGION_IP_PREFIX.keys())

DEVICES = [
    "Chrome_Win10", "Chrome_Mac", "Firefox_Win10", "Safari_Mac",
    "Chrome_Android", "Safari_iOS", "Edge_Win11", "Chrome_Linux",
]

KYC_STATUSES = ["pending", "verified", "under_review", "rejected", "resubmitted"]

# Solo anomaly types (assigned individually per user)
SOLO_ANOMALY_TYPES = [
    "multi_ip_login",
    "unusual_time_login",
    "device_mismatch",
    "brute_force_login",
    "large_withdrawal_dormant",
    "deposit_withdraw_abuse",
    "structuring",
    "volume_spike",
    "single_instrument",
    "latency_arbitrage",
    "consistent_profit",
    "bot_behavior",
    "device_switching",
    "rapid_kyc_withdrawal",
]


# ============================================================================
# HELPERS
# ============================================================================

rng = np.random.default_rng(SEED)


def make_ip(region: str) -> str:
    """Generate a plausible IP address for the given geographic region."""
    prefix = REGION_IP_PREFIX[region]
    parts = prefix.split(".")
    while len(parts) < 4:
        parts.append(str(rng.integers(1, 255)))
    return ".".join(parts)


def make_fingerprint(device: str, user_id: str) -> str:
    """Deterministic 12-char device fingerprint from device + user."""
    raw = f"{device}_{user_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def rand_ts(day_offset: int, hour_lo: int = 8, hour_hi: int = 22) -> datetime:
    """Generate a random timestamp on a specific day within an hour range."""
    h = int(rng.integers(hour_lo, hour_hi))
    m = int(rng.integers(0, 60))
    s = int(rng.integers(0, 60))
    return START + timedelta(days=int(day_offset), hours=h, minutes=m, seconds=s)


def ts_offset(base: datetime, minutes: float) -> datetime:
    """Offset a timestamp by fractional minutes."""
    return base + timedelta(minutes=minutes)


# ============================================================================
# USER PROFILE GENERATION
# ============================================================================

def create_users() -> list[dict]:
    """
    Create 500 user profiles. ~15% are flagged anomalous with 1-2 solo
    anomaly types each. Group anomalies (ip_hub, collusion_ring) are
    layered on top afterward.
    """
    n_anom = int(NUM_USERS * ANOMALY_RATIO)
    n_norm = NUM_USERS - n_anom
    users = []

    for i in range(NUM_USERS):
        uid = f"user_{i + 1:04d}"
        region = str(rng.choice(REGIONS))
        device = str(rng.choice(DEVICES))
        ip = make_ip(region)

        if i < n_norm:
            users.append(dict(
                user_id=uid, home_region=region, primary_ip=ip,
                primary_device=device, is_anomaly=0, anomaly_types=set(),
            ))
        else:
            n_types = int(rng.choice([1, 2], p=[0.55, 0.45]))
            types = set(rng.choice(SOLO_ANOMALY_TYPES, size=n_types, replace=False))
            users.append(dict(
                user_id=uid, home_region=region, primary_ip=ip,
                primary_device=device, is_anomaly=1, anomaly_types=types,
            ))

    # ── Group anomaly: IP Hub (8 users share a single IP) ────────────
    hub_ip = make_ip("Singapore")
    anom_indices = list(range(n_norm, NUM_USERS))
    hub_idxs = rng.choice(anom_indices, size=min(8, len(anom_indices)), replace=False)
    for idx in hub_idxs:
        users[idx]["primary_ip"] = hub_ip
        users[idx]["anomaly_types"].add("ip_hub")

    # ── Group anomaly: Collusion Ring (2 groups of 3 users) ──────────
    remaining = [i for i in anom_indices if i not in hub_idxs]
    if len(remaining) >= 6:
        ring_idxs = rng.choice(remaining, size=6, replace=False)
        for idx in ring_idxs:
            users[idx]["anomaly_types"].add("collusion_ring")

    # Shuffle so anomalous users are distributed throughout the dataset
    rng.shuffle(users)
    return users


# ============================================================================
# CLIENT PORTAL EVENT GENERATION
# ============================================================================

def generate_portal_events(users: list[dict]) -> pd.DataFrame:
    """
    Generate client portal events: logins, logouts, deposits, withdrawals,
    KYC changes, support tickets, account mods, document uploads.

    Anomaly patterns are injected inline per-user based on their anomaly_types.
    """
    rows = []
    eid = 0

    for u in users:
        uid = u["user_id"]
        region = u["home_region"]
        ip0 = u["primary_ip"]
        dev0 = u["primary_device"]
        is_anom = u["is_anomaly"]
        atypes = u["anomaly_types"]
        atype_str = "|".join(sorted(atypes)) if atypes else ""

        # ── Activity calendar ─────────────────────────────────────────
        n_days = int(rng.integers(20, 45) if is_anom else rng.integers(12, 30))
        active_days = sorted(rng.choice(WINDOW_DAYS, size=min(n_days, WINDOW_DAYS), replace=False))
        active_days = [int(d) for d in active_days]

        # ── Anomaly-specific day injection ────────────────────────────

        # large_withdrawal_dormant: force a gap then a late-window withdrawal
        dormant_day = None
        if "large_withdrawal_dormant" in atypes:
            dormant_day = int(rng.integers(65, 85))
            # Remove activity between day 30 and dormant_day to create gap
            active_days = [d for d in active_days if d < 30 or d >= dormant_day]
            if dormant_day not in active_days:
                active_days.append(dormant_day)
            active_days = sorted(set(active_days))

        # deposit_withdraw_abuse: single-day deposit→tiny-trade→withdraw cycle
        abuse_day = None
        if "deposit_withdraw_abuse" in atypes:
            abuse_day = int(rng.integers(40, 70))
            if abuse_day not in active_days:
                active_days.append(abuse_day)
            active_days = sorted(set(active_days))

        # rapid_kyc_withdrawal: KYC change immediately before withdrawal
        kyc_wd_day = None
        if "rapid_kyc_withdrawal" in atypes:
            kyc_wd_day = int(rng.integers(50, 80))
            if kyc_wd_day not in active_days:
                active_days.append(kyc_wd_day)
            active_days = sorted(set(active_days))

        # structuring: add extra days for many small deposits
        if "structuring" in atypes:
            extra = sorted(rng.choice(WINDOW_DAYS, size=12, replace=False))
            active_days = sorted(set(active_days) | set(int(d) for d in extra))

        # ── Generate events per active day ────────────────────────────
        for day in active_days:

            # ─── IP selection ─────────────────────────────────────────
            if "multi_ip_login" in atypes:
                ip = make_ip(str(rng.choice(REGIONS)))
                geo = str(rng.choice(REGIONS))
            elif "ip_hub" in atypes:
                ip = ip0  # shared hub IP
                geo = region
            else:
                ip = ip0 if rng.random() > 0.05 else make_ip(region)
                geo = region

            # ─── Device selection ─────────────────────────────────────
            if "device_mismatch" in atypes or "device_switching" in atypes:
                dev = str(rng.choice(DEVICES))
            else:
                dev = dev0 if rng.random() > 0.08 else str(rng.choice(DEVICES))

            fp = make_fingerprint(dev, uid)

            # ─── Login hour ───────────────────────────────────────────
            if "unusual_time_login" in atypes and rng.random() < 0.45:
                hour_lo, hour_hi = 1, 5
            else:
                hour_lo, hour_hi = 8, 22

            login_ts = rand_ts(day, hour_lo, hour_hi)

            # ─── Session duration & page views ────────────────────────
            if "bot_behavior" in atypes:
                sess_dur = round(float(rng.uniform(0.2, 1.5)), 1)
                pages = int(rng.integers(25, 80))  # rapid automated navigation
            else:
                sess_dur = round(min(float(rng.lognormal(2.5, 0.8)), 120.0), 1)
                pages = int(rng.integers(3, 18))

            # Base event template for this session
            base = dict(
                user_id=uid, ip_address=ip, device_fingerprint=fp,
                geo_location=geo, session_duration_min=0.0,
                amount=0.0, currency="USD", login_status="",
                kyc_status="", page_views=0,
                is_anomaly=is_anom, anomaly_type=atype_str,
            )

            # ─── BRUTE FORCE: cluster of failed logins before success ─
            if "brute_force_login" in atypes and rng.random() < 0.20:
                n_fail = int(rng.integers(3, 8))
                for f in range(n_fail):
                    eid += 1
                    fail_ts = login_ts - timedelta(
                        seconds=int(rng.integers(8, 25)) * (n_fail - f)
                    )
                    rows.append({
                        **base,
                        "event_id": f"PE_{eid:06d}",
                        "timestamp": fail_ts,
                        "event_type": "login",
                        "login_status": "failed",
                    })

            # ─── SUCCESSFUL LOGIN ─────────────────────────────────────
            eid += 1
            rows.append({
                **base,
                "event_id": f"PE_{eid:06d}",
                "timestamp": login_ts,
                "event_type": "login",
                "login_status": "success",
                "session_duration_min": sess_dur,
                "page_views": pages,
            })

            # ─── LOGOUT ──────────────────────────────────────────────
            eid += 1
            rows.append({
                **base,
                "event_id": f"PE_{eid:06d}",
                "timestamp": ts_offset(login_ts, sess_dur),
                "event_type": "logout",
            })

            # ─── NORMAL DEPOSIT (~12% chance) ─────────────────────────
            if rng.random() < 0.12:
                eid += 1
                amt = round(min(float(rng.lognormal(6.5, 1.0)), 15000.0), 2)
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(login_ts, float(rng.uniform(1, max(2, sess_dur)))),
                    "event_type": "deposit",
                    "amount": amt,
                })

            # ─── NORMAL WITHDRAWAL (~8% chance) ───────────────────────
            if rng.random() < 0.08:
                eid += 1
                amt = round(min(float(rng.lognormal(6.0, 1.0)), 10000.0), 2)
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(login_ts, float(rng.uniform(1, max(2, sess_dur)))),
                    "event_type": "withdrawal",
                    "amount": amt,
                })

            # ─── STRUCTURING: high-frequency small deposits ───────────
            if "structuring" in atypes and rng.random() < 0.55:
                n_micro = int(rng.integers(2, 6))
                for _ in range(n_micro):
                    eid += 1
                    # Amounts clustered just under common thresholds
                    amt = round(float(rng.uniform(150, 950)), 2)
                    rows.append({
                        **base,
                        "event_id": f"PE_{eid:06d}",
                        "timestamp": ts_offset(
                            login_ts, float(rng.uniform(0.5, max(1, sess_dur)))
                        ),
                        "event_type": "deposit",
                        "amount": amt,
                    })

            # ─── LARGE WITHDRAWAL AFTER DORMANCY ──────────────────────
            if dormant_day is not None and day == dormant_day:
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(login_ts, 2.0),
                    "event_type": "withdrawal",
                    "amount": round(float(rng.uniform(8000, 30000)), 2),
                })

            # ─── DEPOSIT → WITHDRAW ABUSE ─────────────────────────────
            if abuse_day is not None and day == abuse_day:
                dep_amt = round(float(rng.uniform(5000, 20000)), 2)
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(login_ts, 1.0),
                    "event_type": "deposit",
                    "amount": dep_amt,
                })
                # Withdrawal of nearly the same amount ~20 min later
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(login_ts, 22.0),
                    "event_type": "withdrawal",
                    "amount": round(dep_amt * float(rng.uniform(0.92, 0.99)), 2),
                })

            # ─── RAPID KYC CHANGE → WITHDRAWAL ───────────────────────
            if kyc_wd_day is not None and day == kyc_wd_day:
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(login_ts, 1.0),
                    "event_type": "kyc_change",
                    "kyc_status": "verified",
                })
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(login_ts, 6.0),
                    "event_type": "withdrawal",
                    "amount": round(float(rng.uniform(5000, 25000)), 2),
                })

            # ─── OCCASIONAL BACKGROUND EVENTS ─────────────────────────
            if rng.random() < 0.04:
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(
                        login_ts, float(rng.uniform(1, max(2, sess_dur)))
                    ),
                    "event_type": "kyc_change",
                    "kyc_status": str(rng.choice(KYC_STATUSES)),
                })

            if rng.random() < 0.05:
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(
                        login_ts, float(rng.uniform(1, max(2, sess_dur)))
                    ),
                    "event_type": "support_ticket",
                })

            if rng.random() < 0.04:
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(
                        login_ts, float(rng.uniform(1, max(2, sess_dur)))
                    ),
                    "event_type": "account_modification",
                })

            if rng.random() < 0.03:
                eid += 1
                rows.append({
                    **base,
                    "event_id": f"PE_{eid:06d}",
                    "timestamp": ts_offset(
                        login_ts, float(rng.uniform(1, max(2, sess_dur)))
                    ),
                    "event_type": "document_upload",
                })

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================================
# TRADING EVENT GENERATION
# ============================================================================

def generate_trading_events(users: list[dict]) -> pd.DataFrame:
    """
    Generate WebTrader/TradingTerminal events: trades with instrument,
    lot size, volume, margin, PnL, and duration.

    Injects: volume_spike, single_instrument, latency_arbitrage,
    consistent_profit, collusion_ring (mirror trades).
    """
    rows = []
    eid = 0

    # ── Pre-identify collusion ring groups ────────────────────────────
    ring_members = [u for u in users if "collusion_ring" in u["anomaly_types"]]
    ring_groups = []
    for i in range(0, len(ring_members), 3):
        grp = ring_members[i : i + 3]
        if len(grp) >= 2:
            ring_groups.append(grp)

    for u in users:
        uid = u["user_id"]
        region = u["home_region"]
        ip0 = u["primary_ip"]
        is_anom = u["is_anomaly"]
        atypes = u["anomaly_types"]
        atype_str = "|".join(sorted(atypes)) if atypes else ""

        # ── Trading calendar ──────────────────────────────────────────
        n_days = int(rng.integers(18, 42) if is_anom else rng.integers(10, 28))
        active_days = sorted(rng.choice(WINDOW_DAYS, size=min(n_days, WINDOW_DAYS), replace=False))
        active_days = [int(d) for d in active_days]

        # Preferred instruments
        if "single_instrument" in atypes:
            preferred = [str(rng.choice(INSTRUMENTS))]
        else:
            n_inst = int(rng.integers(2, 5))
            preferred = [str(x) for x in rng.choice(INSTRUMENTS, size=n_inst, replace=False)]

        # Baseline lot size for this user
        base_lot = float(rng.uniform(0.01, 0.5))

        # ── Volume spike: 10x activity in a 2-day burst ──────────────
        spike_days = set()
        if "volume_spike" in atypes:
            spike_start = int(rng.integers(40, 80))
            spike_days = {spike_start, spike_start + 1}
            for d in spike_days:
                if d not in active_days:
                    active_days.append(d)
            active_days = sorted(set(active_days))

        # ── Deposit-withdraw abuse: generate only 1-2 tiny trades ─────
        abuse_day = None
        if "deposit_withdraw_abuse" in atypes:
            abuse_day = int(rng.integers(40, 70))
            if abuse_day not in active_days:
                active_days.append(abuse_day)
            active_days = sorted(set(active_days))

        for day in active_days:
            day = int(day)

            # Trades per day
            if day in spike_days:
                n_trades = int(rng.integers(15, 35))  # 10x spike
            elif "latency_arbitrage" in atypes and rng.random() < 0.35:
                n_trades = int(rng.integers(10, 25))  # high-frequency burst
            elif abuse_day is not None and day == abuse_day:
                n_trades = int(rng.integers(1, 3))  # minimal trading
            else:
                n_trades = int(rng.integers(1, 5))

            for _ in range(n_trades):
                eid += 1
                ts = rand_ts(day)

                # Instrument selection
                if "single_instrument" in atypes:
                    inst = preferred[0]
                else:
                    inst = (
                        str(rng.choice(preferred))
                        if rng.random() < 0.80
                        else str(rng.choice(INSTRUMENTS))
                    )

                direction = str(rng.choice(["buy", "sell"]))

                # Lot size
                if day in spike_days:
                    lot = round(float(base_lot * rng.uniform(5, 15)), 2)
                elif abuse_day is not None and day == abuse_day:
                    lot = round(float(rng.uniform(0.01, 0.05)), 2)  # tiny trades
                else:
                    lot = round(float(base_lot * rng.uniform(0.5, 2.0)), 2)
                lot = max(0.01, lot)

                volume = round(lot * 100_000, 2)  # Standard lot = 100K units
                margin = round(float(rng.uniform(5, 65)), 1)

                # PnL
                if "consistent_profit" in atypes:
                    # Abnormally high win rate (~90%)
                    if rng.random() < 0.90:
                        pnl = round(float(rng.uniform(10, 250)), 2)
                    else:
                        pnl = -round(float(rng.uniform(2, 30)), 2)
                elif "latency_arbitrage" in atypes:
                    # Small consistent profits from ultra-fast trades
                    pnl = round(float(rng.uniform(2, 60)), 2)
                elif abuse_day is not None and day == abuse_day:
                    # Minimal PnL for abuse pattern
                    pnl = round(float(rng.normal(0, 5)), 2)
                else:
                    pnl = round(float(rng.normal(0, lot * 50)), 2)

                # Trade duration
                if "latency_arbitrage" in atypes:
                    trade_dur = round(float(rng.uniform(0.3, 5.0)), 1)  # sub-5s
                else:
                    trade_dur = round(
                        min(float(rng.lognormal(5.0, 1.5)), 86400.0), 1
                    )

                # IP for trading session
                if "multi_ip_login" in atypes:
                    ip = make_ip(str(rng.choice(REGIONS)))
                else:
                    ip = ip0 if rng.random() > 0.05 else make_ip(region)

                rows.append({
                    "event_id": f"TE_{eid:06d}",
                    "timestamp": ts,
                    "user_id": uid,
                    "instrument": inst,
                    "direction": direction,
                    "lot_size": lot,
                    "trade_volume_usd": volume,
                    "margin_used_pct": margin,
                    "pnl": pnl,
                    "trade_duration_sec": trade_dur,
                    "ip_address": ip,
                    "is_anomaly": is_anom,
                    "anomaly_type": atype_str,
                })

    # ── COLLUSION RING: synchronized mirror trades ────────────────────
    for group in ring_groups:
        n_ring_trades = int(rng.integers(20, 40))
        ring_days = sorted(rng.choice(WINDOW_DAYS, size=n_ring_trades, replace=False))

        for day in ring_days:
            base_ts = rand_ts(int(day))
            inst = str(rng.choice(INSTRUMENTS))
            lot = round(float(rng.uniform(0.1, 1.5)), 2)
            primary_dir = str(rng.choice(["buy", "sell"]))
            mirror_dir = "sell" if primary_dir == "buy" else "buy"

            for i, member in enumerate(group):
                eid += 1
                d = primary_dir if i == 0 else mirror_dir
                pnl_sign = 1.0 if i == 0 else -1.0

                rows.append({
                    "event_id": f"TE_{eid:06d}",
                    # Slight time offset (0-8s) to simulate near-simultaneous
                    "timestamp": base_ts + timedelta(seconds=int(rng.integers(0, 9))),
                    "user_id": member["user_id"],
                    "instrument": inst,
                    "direction": d,
                    "lot_size": lot,
                    "trade_volume_usd": round(lot * 100_000, 2),
                    "margin_used_pct": round(float(rng.uniform(10, 40)), 1),
                    "pnl": round(pnl_sign * float(rng.uniform(10, 120)), 2),
                    "trade_duration_sec": round(float(rng.uniform(30, 300)), 1),
                    "ip_address": member["primary_ip"],
                    "is_anomaly": 1,
                    "anomaly_type": "|".join(sorted(member["anomaly_types"])),
                })

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================================
# GROUND TRUTH (EVALUATION ONLY)
# ============================================================================

def build_ground_truth(users: list[dict]) -> pd.DataFrame:
    """
    Separate ground truth file mapping user_id → anomaly labels.
    This MUST be stripped before feature engineering (Phase 2).
    Used ONLY for final evaluation metrics in Phase 3.
    """
    return pd.DataFrame([
        {
            "user_id": u["user_id"],
            "is_anomaly": u["is_anomaly"],
            "anomaly_types": "|".join(sorted(u["anomaly_types"])) if u["anomaly_types"] else "",
            "home_region": u["home_region"],
        }
        for u in users
    ])


# ============================================================================
# MAIN
# ============================================================================

def main():
    out_dir = Path(__file__).resolve().parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ForexGuard — Synthetic Data Generator")
    print("=" * 60)

    # ── Step 1: Users ─────────────────────────────────────────────────
    print("\n[1/4] Generating user profiles...")
    users = create_users()
    n_anom = sum(1 for u in users if u["is_anomaly"])
    print(f"  Total users : {len(users)}")
    print(f"  Normal      : {len(users) - n_anom}")
    print(f"  Anomalous   : {n_anom} ({n_anom / len(users) * 100:.1f}%)")

    # ── Step 2: Portal Events ─────────────────────────────────────────
    print("\n[2/4] Generating client portal events...")
    portal_df = generate_portal_events(users)
    print(f"  Total events: {len(portal_df):,}")
    print(f"  Event types :")
    for etype, count in portal_df["event_type"].value_counts().items():
        print(f"    {etype:25s} {count:>6,}")

    # ── Step 3: Trading Events ────────────────────────────────────────
    print("\n[3/4] Generating trading events...")
    trading_df = generate_trading_events(users)
    print(f"  Total events: {len(trading_df):,}")
    print(f"  Instruments : {trading_df['instrument'].nunique()}")
    print(f"  Unique users: {trading_df['user_id'].nunique()}")

    # ── Step 4: Ground Truth & Save ───────────────────────────────────
    print("\n[4/4] Building ground truth & saving...")
    gt_df = build_ground_truth(users)

    portal_df.to_csv(out_dir / "client_portal_events.csv", index=False)
    trading_df.to_csv(out_dir / "trading_events.csv", index=False)
    gt_df.to_csv(out_dir / "ground_truth.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────
    total = len(portal_df) + len(trading_df)
    print(f"\n{'=' * 60}")
    print(f"  TOTAL EVENTS: {total:,}")
    print(f"  Portal      : {len(portal_df):,}")
    print(f"  Trading     : {len(trading_df):,}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'=' * 60}")

    # Anomaly distribution
    print("\n  Anomaly Type Distribution (across users):")
    type_counts: Counter = Counter()
    for u in users:
        for t in u["anomaly_types"]:
            type_counts[t] += 1
    for atype, count in type_counts.most_common():
        print(f"    {atype:30s} {count:>3} users")

    # Anomaly event counts
    portal_anom = portal_df[portal_df["is_anomaly"] == 1]
    trade_anom = trading_df[trading_df["is_anomaly"] == 1]
    print(f"\n  Anomalous events:")
    print(f"    Portal  : {len(portal_anom):,} / {len(portal_df):,} "
          f"({len(portal_anom) / len(portal_df) * 100:.1f}%)")
    print(f"    Trading : {len(trade_anom):,} / {len(trading_df):,} "
          f"({len(trade_anom) / len(trading_df) * 100:.1f}%)")
    print()


if __name__ == "__main__":
    main()
