#!/usr/bin/env python3
"""
ForexGuard -- Async Streaming Simulator
=========================================
Replays raw events from the synthetic dataset against the running FastAPI
/score endpoint, simulating a real-time Kafka-like event stream.

Usage:
  1. Start the API:   uvicorn src.api.main:app --port 8000
  2. Run simulator:   python src/streaming/simulator.py

Options (env vars):
  API_URL       Base URL (default: http://localhost:8000)
  BATCH_SIZE    Concurrent requests per batch (default: 20)
  MAX_EVENTS    Stop after N events (default: 0 = all)
  DELAY_MS      Delay between batches in ms (default: 0)
"""

import os
import sys
import time
import asyncio
import math
import numpy as np
import pandas as pd
import httpx
from pathlib import Path
from dataclasses import dataclass, field

# ============================================================================
# Configuration
# ============================================================================

API_URL = os.environ.get("API_URL", "http://localhost:8000")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "20"))
MAX_EVENTS = int(os.environ.get("MAX_EVENTS", "0"))  # 0 = all
DELAY_MS = int(os.environ.get("DELAY_MS", "0"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ============================================================================
# Stats tracker
# ============================================================================

@dataclass
class Stats:
    total_sent: int = 0
    total_ok: int = 0
    total_errors: int = 0
    total_anomalies: int = 0
    anomaly_users: dict = field(default_factory=dict)
    start_time: float = 0.0

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def eps(self) -> float:
        e = self.elapsed()
        return self.total_ok / e if e > 0 else 0.0


stats = Stats()


# ============================================================================
# Terminal formatting
# ============================================================================

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header():
    print(f"\n{BOLD}{'='*72}")
    print(f"  ForexGuard -- Streaming Simulator")
    print(f"  Target : {API_URL}/score")
    print(f"  Batch  : {BATCH_SIZE} concurrent requests")
    print(f"{'='*72}{RESET}\n")


def print_progress(i: int, total: int):
    pct = i / total * 100
    bar_len = 30
    filled = int(bar_len * i // total)
    bar = "=" * filled + "-" * (bar_len - filled)
    eps = stats.eps()
    sys.stdout.write(
        f"\r  {DIM}[{bar}] {i:>6,}/{total:,} ({pct:5.1f}%)  "
        f"{eps:6.0f} events/s  "
        f"anomalies: {RED}{stats.total_anomalies}{RESET}{DIM}  "
        f"errors: {stats.total_errors}{RESET}"
    )
    sys.stdout.flush()


def print_anomaly(resp: dict):
    uid = resp["user_id"]
    score = resp["risk_score"]
    eid = resp["event_id"]
    contributors = resp.get("top_contributors", [])

    # Track per-user anomaly count
    stats.anomaly_users[uid] = stats.anomaly_users.get(uid, 0) + 1
    count = stats.anomaly_users[uid]

    print(f"\n  {RED}{BOLD}!! ANOMALY DETECTED{RESET}  "
          f"{YELLOW}{uid}{RESET}  "
          f"score={RED}{score:.4f}{RESET}  "
          f"event={DIM}{eid}{RESET}  "
          f"(alert #{count} for this user)")

    if contributors:
        for c in contributors:
            print(f"     {CYAN}-> {c['feature']}{RESET}: error={c['error']:.4f}")


def print_summary(total: int):
    elapsed = stats.elapsed()
    print(f"\n\n{BOLD}{'='*72}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'='*72}{RESET}")
    print(f"  Events sent     : {stats.total_sent:,}")
    print(f"  Successful      : {GREEN}{stats.total_ok:,}{RESET}")
    print(f"  Errors          : {RED}{stats.total_errors:,}{RESET}")
    print(f"  Anomalies found : {RED}{BOLD}{stats.total_anomalies:,}{RESET}")
    print(f"  Unique flagged  : {len(stats.anomaly_users)} users")
    print(f"  Duration        : {elapsed:.1f}s")
    print(f"  Throughput      : {stats.eps():,.0f} events/s")

    if stats.anomaly_users:
        print(f"\n  {BOLD}Top flagged users:{RESET}")
        sorted_users = sorted(stats.anomaly_users.items(), key=lambda x: -x[1])
        for uid, cnt in sorted_users[:10]:
            bar = "=" * min(cnt, 40)
            print(f"    {YELLOW}{uid}{RESET}  {cnt:>3} alerts  {RED}{bar}{RESET}")


# ============================================================================
# Event loading
# ============================================================================

def load_events() -> list[dict]:
    """Load, merge, strip labels, sort chronologically."""
    raw_dir = PROJECT_ROOT / "data" / "raw"

    print(f"  Loading portal events...  ", end="", flush=True)
    portal = pd.read_csv(raw_dir / "client_portal_events.csv")
    print(f"{len(portal):,} rows")

    print(f"  Loading trading events... ", end="", flush=True)
    trading = pd.read_csv(raw_dir / "trading_events.csv")
    print(f"{len(trading):,} rows")

    # Strip labels (unsupervised discipline)
    for col in ("is_anomaly", "anomaly_type"):
        portal.drop(columns=[col], inplace=True, errors="ignore")
        trading.drop(columns=[col], inplace=True, errors="ignore")

    # Merge and sort
    all_events = pd.concat([portal, trading], ignore_index=True)
    all_events.sort_values("timestamp", inplace=True)

    # Convert to JSON-safe native Python types (no numpy int64/float64)
    records = []
    for _, row in all_events.iterrows():
        rec = {}
        for k, v in row.items():
            if pd.isna(v):
                continue  # skip NaN fields entirely
            elif isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = float(v)
            else:
                rec[k] = v
        records.append(rec)

    print(f"  Merged & sorted:          {len(records):,} events")
    return records


# ============================================================================
# Async scoring
# ============================================================================

async def score_one(
    client: httpx.AsyncClient,
    event: dict,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Send a single event to /score and return the response."""
    async with semaphore:
        try:
            # Filter out None values to keep payload clean
            payload = {k: v for k, v in event.items() if v is not None}
            resp = await client.post(f"{API_URL}/score", json=payload, timeout=10.0)
            resp.raise_for_status()
            stats.total_ok += 1
            return resp.json()
        except Exception:
            stats.total_errors += 1
            return None


async def run_simulation():
    """Main simulation loop."""
    print_header()

    # Load events
    events = load_events()
    total = len(events)
    if MAX_EVENTS > 0:
        events = events[:MAX_EVENTS]
        total = len(events)
        print(f"  (Limited to first {MAX_EVENTS:,} events)")

    print(f"\n  Streaming {total:,} events -> {API_URL}/score\n")

    # Check API health first
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{API_URL}/health", timeout=5.0)
            health.raise_for_status()
            print(f"  {GREEN}API is healthy{RESET}: {health.json()}\n")
        except Exception as e:
            print(f"  {RED}API not reachable at {API_URL}: {e}{RESET}")
            print(f"  Start the API first: uvicorn src.api.main:app --port 8000")
            return

    # Stream events in batches
    stats.start_time = time.time()
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    async with httpx.AsyncClient() as client:
        n_batches = math.ceil(total / BATCH_SIZE)

        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, total)
            batch = events[start:end]

            # Fire batch concurrently
            tasks = [score_one(client, ev, semaphore) for ev in batch]
            results = await asyncio.gather(*tasks)

            # Process results
            for resp in results:
                stats.total_sent += 1
                if resp and resp.get("is_anomalous"):
                    stats.total_anomalies += 1
                    print_anomaly(resp)

            print_progress(end, total)

            # Optional delay between batches
            if DELAY_MS > 0:
                await asyncio.sleep(DELAY_MS / 1000.0)

    print_summary(total)


# ============================================================================
# Entry point
# ============================================================================

def main():
    asyncio.run(run_simulation())


if __name__ == "__main__":
    main()
