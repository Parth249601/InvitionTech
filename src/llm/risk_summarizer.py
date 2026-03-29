#!/usr/bin/env python3
"""
LLM-Powered Risk Summary Generator
====================================
Generates human-readable compliance narratives for anomalous users by
feeding anomaly detection results into an LLM.

Supports two backends:
  1. **Google Gemini** (default) -- free tier, no credit card needed
  2. **Template fallback** -- deterministic, works offline with zero config

Set the env var GEMINI_API_KEY to enable LLM mode; otherwise the template
engine produces structured narratives from the feature data alone.

Usage:
    summarizer = RiskSummarizer()           # auto-detects backend
    summary = await summarizer.generate(
        user_id="user_042",
        risk_score=41.97,
        top_contributors=[{"feature": "unusual_hour_ratio", "error": 296.86}, ...],
        user_features={"unusual_hour_ratio": 0.44, "unique_ip_count": 32, ...},
    )
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger("forexguard.llm")

# ---------------------------------------------------------------------------
# Feature metadata -- maps internal feature names to human-readable context
# ---------------------------------------------------------------------------
FEATURE_DESCRIPTIONS: dict[str, dict] = {
    # Login & Access
    "login_count": {"label": "Login Frequency", "unit": "logins", "category": "Login & Access"},
    "failed_login_count": {"label": "Failed Login Attempts", "unit": "attempts", "category": "Login & Access"},
    "failed_login_ratio": {"label": "Failed Login Ratio", "unit": "ratio (0-1)", "category": "Login & Access"},
    "max_failed_streak": {"label": "Max Consecutive Failed Logins", "unit": "streak", "category": "Login & Access"},
    "unique_ip_count": {"label": "Unique IP Addresses Used", "unit": "IPs", "category": "Login & Access"},
    "unique_device_count": {"label": "Unique Devices Used", "unit": "devices", "category": "Login & Access"},
    "unique_geo_count": {"label": "Unique Geolocations", "unit": "locations", "category": "Login & Access"},
    "ip_change_rate": {"label": "IP Change Rate", "unit": "changes/login", "category": "Login & Access"},
    "device_change_rate": {"label": "Device Change Rate", "unit": "changes/login", "category": "Login & Access"},
    "unusual_hour_ratio": {"label": "After-Hours Activity Ratio (1-5 AM)", "unit": "ratio (0-1)", "category": "Login & Access"},
    # Session & Behavioral
    "avg_session_duration": {"label": "Average Session Duration", "unit": "minutes", "category": "Behavioral"},
    "std_session_duration": {"label": "Session Duration Variability", "unit": "minutes (std)", "category": "Behavioral"},
    "avg_page_views": {"label": "Average Page Views per Session", "unit": "pages", "category": "Behavioral"},
    "page_view_velocity": {"label": "Page View Velocity", "unit": "pages/min", "category": "Behavioral"},
    # Financial
    "deposit_count": {"label": "Deposit Count", "unit": "deposits", "category": "Financial"},
    "total_deposit_amt": {"label": "Total Deposit Amount", "unit": "USD", "category": "Financial"},
    "avg_deposit_amt": {"label": "Average Deposit Amount", "unit": "USD", "category": "Financial"},
    "std_deposit_amt": {"label": "Deposit Amount Variability", "unit": "USD (std)", "category": "Financial"},
    "small_deposit_ratio": {"label": "Small Deposit Ratio", "unit": "ratio (0-1)", "category": "Financial"},
    "withdrawal_count": {"label": "Withdrawal Count", "unit": "withdrawals", "category": "Financial"},
    "total_withdrawal_amt": {"label": "Total Withdrawal Amount", "unit": "USD", "category": "Financial"},
    "max_single_withdrawal": {"label": "Largest Single Withdrawal", "unit": "USD", "category": "Financial"},
    "net_deposit_flow": {"label": "Net Deposit Flow (deposits - withdrawals)", "unit": "USD", "category": "Financial"},
    "kyc_change_count": {"label": "KYC Status Changes", "unit": "changes", "category": "Account Risk"},
    # Trading
    "trade_count": {"label": "Trade Count", "unit": "trades", "category": "Trading"},
    "avg_lot_size": {"label": "Average Lot Size", "unit": "lots", "category": "Trading"},
    "std_lot_size": {"label": "Lot Size Variability", "unit": "lots (std)", "category": "Trading"},
    "max_lot_size": {"label": "Maximum Lot Size", "unit": "lots", "category": "Trading"},
    "avg_trade_volume": {"label": "Average Trade Volume", "unit": "USD", "category": "Trading"},
    "std_trade_volume": {"label": "Trade Volume Variability", "unit": "USD (std)", "category": "Trading"},
    "volume_spike_ratio": {"label": "Volume Spike Ratio (max/mean)", "unit": "ratio", "category": "Trading"},
    "avg_margin_used": {"label": "Average Margin Usage", "unit": "%", "category": "Trading"},
    "avg_pnl": {"label": "Average P&L", "unit": "USD", "category": "Trading"},
    "std_pnl": {"label": "P&L Variability", "unit": "USD (std)", "category": "Trading"},
    "pnl_win_rate": {"label": "Win Rate (profitable trades)", "unit": "ratio (0-1)", "category": "Trading"},
    "avg_trade_duration": {"label": "Average Trade Duration", "unit": "seconds", "category": "Trading"},
    "min_trade_duration": {"label": "Minimum Trade Duration", "unit": "seconds", "category": "Trading"},
    "short_trade_ratio": {"label": "Ultra-Short Trade Ratio (<5s)", "unit": "ratio (0-1)", "category": "Trading"},
    "instrument_count": {"label": "Unique Instruments Traded", "unit": "instruments", "category": "Trading"},
    "instrument_concentration": {"label": "Instrument Concentration", "unit": "ratio (0-1)", "category": "Trading"},
    "direction_imbalance": {"label": "Buy/Sell Direction Imbalance", "unit": "ratio (0-1)", "category": "Trading"},
    "avg_inter_trade_time": {"label": "Average Time Between Trades", "unit": "seconds", "category": "Trading"},
    # Cross-domain
    "avg_inter_event_time": {"label": "Average Time Between Events", "unit": "seconds", "category": "Temporal"},
    "std_inter_event_time": {"label": "Event Timing Variability", "unit": "seconds (std)", "category": "Temporal"},
    "min_inter_event_time": {"label": "Minimum Inter-Event Time", "unit": "seconds", "category": "Temporal"},
    "activity_hour_entropy": {"label": "Activity Hour Entropy", "unit": "bits", "category": "Temporal"},
    "dormancy_max_days": {"label": "Maximum Dormancy Period", "unit": "days", "category": "Temporal"},
    "total_event_count": {"label": "Total Event Count", "unit": "events", "category": "Temporal"},
    "deposit_to_trade_ratio": {"label": "Deposit-to-Trade Ratio", "unit": "ratio", "category": "Financial"},
    "kyc_to_withdrawal_hours": {"label": "Time from KYC Change to Withdrawal", "unit": "hours", "category": "Account Risk"},
}

# ---------------------------------------------------------------------------
# Risk level classification
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score >= 20.0:
        return "CRITICAL"
    if score >= 5.0:
        return "HIGH"
    if score >= 1.5:
        return "MEDIUM"
    return "LOW"


def _format_value(feature: str, value: float) -> str:
    """Format a feature value with its unit for display."""
    meta = FEATURE_DESCRIPTIONS.get(feature, {})
    unit = meta.get("unit", "")
    if "USD" in unit:
        return f"${value:,.2f}"
    if "ratio" in unit:
        return f"{value:.2%}" if value <= 1.0 else f"{value:.2f}"
    if "minutes" in unit:
        return f"{value:.1f} min"
    if "seconds" in unit:
        return f"{value:.1f}s"
    if "days" in unit:
        return f"{value:.1f} days"
    if "hours" in unit:
        return f"{value:.1f} hours"
    return f"{value:,.2f} {unit}".strip()


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(
    user_id: str,
    risk_score: float,
    top_contributors: list[dict],
    user_features: dict[str, float],
) -> str:
    """Build a structured prompt for the LLM."""
    level = _risk_level(risk_score)

    # Build contributor detail
    contributor_lines = []
    for c in top_contributors:
        feat = c["feature"]
        meta = FEATURE_DESCRIPTIONS.get(feat, {"label": feat, "category": "Unknown"})
        val = user_features.get(feat, 0.0)
        contributor_lines.append(
            f"  - {meta['label']} ({feat}): value={_format_value(feat, val)}, "
            f"reconstruction_error={c['error']:.2f}, category={meta['category']}"
        )

    # Build full feature snapshot grouped by category
    categories: dict[str, list[str]] = {}
    for feat, val in user_features.items():
        meta = FEATURE_DESCRIPTIONS.get(feat, {"label": feat, "category": "Other"})
        cat = meta["category"]
        categories.setdefault(cat, []).append(
            f"    {meta['label']}: {_format_value(feat, val)}"
        )

    snapshot_lines = []
    for cat in sorted(categories):
        snapshot_lines.append(f"  [{cat}]")
        snapshot_lines.extend(categories[cat])

    prompt = f"""You are a compliance risk analyst AI for a forex brokerage. Generate a concise, actionable risk summary for the compliance team based on the anomaly detection results below.

## Anomaly Detection Results
- **User ID**: {user_id}
- **Risk Score**: {risk_score:.4f} (threshold: 0.726)
- **Risk Level**: {level}
- **Top Contributing Anomalous Features**:
{chr(10).join(contributor_lines)}

## Full User Behavioral Profile
{chr(10).join(snapshot_lines)}

## Instructions
Write a 3-5 sentence risk summary that:
1. States the risk level and primary concern in plain language
2. Explains which specific behaviors are suspicious and WHY they indicate potential fraud/abuse
3. Connects the anomalous features to known forex fraud patterns (e.g., account takeover, money laundering structuring, bonus abuse, latency arbitrage, collusion)
4. Recommends a specific next action for the compliance team (e.g., freeze account, escalate to fraud team, request additional KYC verification, monitor closely)

Be specific with numbers. Do not use jargon without explanation. Write for a human compliance officer who needs to decide what to do next."""

    return prompt


# ---------------------------------------------------------------------------
# Template-based fallback summarizer
# ---------------------------------------------------------------------------

_PATTERN_RULES: list[tuple[str, str, str]] = [
    # (feature, condition_description, fraud_pattern)
    ("unusual_hour_ratio", "concentrated activity during 1-5 AM hours", "potential account takeover or unauthorized access from a different timezone"),
    ("unique_ip_count", "abnormally high number of unique IP addresses", "possible credential sharing, account takeover, or use of VPN/proxy rotation"),
    ("max_failed_streak", "multiple consecutive failed login attempts", "brute-force attack or credential stuffing attempt"),
    ("page_view_velocity", "extremely rapid page navigation", "bot-like automated activity or scripted access"),
    ("small_deposit_ratio", "high proportion of small deposits", "potential structuring to avoid reporting thresholds (anti-money laundering concern)"),
    ("max_single_withdrawal", "unusually large single withdrawal", "potential account drain following compromise or money laundering cashout"),
    ("net_deposit_flow", "significant negative deposit flow", "funds being extracted — possible abuse or compromised account"),
    ("kyc_to_withdrawal_hours", "very short gap between KYC change and withdrawal", "possible identity manipulation to facilitate unauthorized withdrawal"),
    ("kyc_change_count", "frequent KYC status changes", "suspicious identity document activity — potential synthetic identity or takeover"),
    ("volume_spike_ratio", "sudden spike in trading volume", "possible wash trading, market manipulation, or compromised account being exploited"),
    ("short_trade_ratio", "extremely high ratio of ultra-short trades (<5s)", "latency arbitrage or algorithmic exploitation pattern"),
    ("instrument_concentration", "trading concentrated on a single instrument", "potential market manipulation or insider trading pattern"),
    ("pnl_win_rate", "abnormally high win rate", "possible exploitation of platform latency or insider information"),
    ("direction_imbalance", "extreme buy/sell imbalance", "one-sided exposure suggesting coordinated or manipulative trading"),
    ("activity_hour_entropy", "highly irregular activity distribution across hours", "non-human activity pattern suggesting automated or multi-timezone access"),
    ("deposit_to_trade_ratio", "unusual deposit-to-trade ratio", "potential bonus abuse or deposit-withdraw cycling without genuine trading"),
    ("unique_device_count", "multiple devices used", "possible account sharing or credential compromise across devices"),
    ("device_change_rate", "frequent device switching", "evasion behavior — attempting to avoid device fingerprinting"),
    ("dormancy_max_days", "long dormancy period followed by sudden activity", "dormant account reactivation — possible compromise or sale of credentials"),
    ("avg_margin_used", "unusually high margin utilization", "aggressive risk-taking possibly linked to account exploitation"),
    ("deposit_count", "unusually high number of deposits", "potential structuring or rapid deposit cycling"),
]


def _generate_template_summary(
    user_id: str,
    risk_score: float,
    top_contributors: list[dict],
    user_features: dict[str, float],
) -> str:
    """Produce a structured risk narrative without an LLM."""
    level = _risk_level(risk_score)
    top_feats = [c["feature"] for c in top_contributors]

    # Match patterns
    matched: list[str] = []
    for feat, description, pattern in _PATTERN_RULES:
        if feat in top_feats:
            val = user_features.get(feat, 0.0)
            meta = FEATURE_DESCRIPTIONS.get(feat, {"label": feat})
            matched.append(
                f"{meta['label']} shows {description} "
                f"(value: {_format_value(feat, val)}), indicating {pattern}."
            )

    if not matched:
        # Generic fallback for unmapped features
        for c in top_contributors[:3]:
            feat = c["feature"]
            meta = FEATURE_DESCRIPTIONS.get(feat, {"label": feat, "category": "Unknown"})
            val = user_features.get(feat, 0.0)
            matched.append(
                f"{meta['label']} ({meta.get('category', 'Unknown')} category) shows "
                f"anomalous value of {_format_value(feat, val)} with high reconstruction error "
                f"({c['error']:.2f}), deviating significantly from normal user behavior."
            )

    # Build narrative
    lines = [
        f"**{level} RISK ALERT** for user `{user_id}` — Risk Score: {risk_score:.2f} "
        f"(threshold: 0.726).",
        "",
    ]
    lines.append("**Key Findings:**")
    for i, m in enumerate(matched[:3], 1):
        lines.append(f"{i}. {m}")

    # Recommendation
    lines.append("")
    if level == "CRITICAL":
        lines.append(
            "**Recommended Action:** Immediately freeze the account and escalate to "
            "the fraud investigation team. Preserve all session logs and transaction "
            "records for forensic review."
        )
    elif level == "HIGH":
        lines.append(
            "**Recommended Action:** Escalate to senior compliance for manual review. "
            "Consider temporary trading restrictions pending investigation. Request "
            "additional identity verification from the user."
        )
    elif level == "MEDIUM":
        lines.append(
            "**Recommended Action:** Flag for enhanced monitoring over the next 48 hours. "
            "If anomalous behavior persists, escalate to compliance review."
        )
    else:
        lines.append(
            "**Recommended Action:** Continue standard monitoring. Log this alert for "
            "trend analysis."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main summarizer class
# ---------------------------------------------------------------------------

class RiskSummarizer:
    """
    Generates LLM-powered or template-based risk narratives.

    Set GEMINI_API_KEY env var to use Google Gemini; otherwise falls back to
    the deterministic template engine.
    """

    def __init__(self) -> None:
        self._gemini_key: Optional[str] = os.environ.get("GEMINI_API_KEY")
        self._client = None
        self._model_name = "gemini-2.0-flash"

        if self._gemini_key:
            try:
                from google import generativeai as genai
                genai.configure(api_key=self._gemini_key)
                self._client = genai.GenerativeModel(self._model_name)
                logger.info("RiskSummarizer: Gemini backend enabled (%s)", self._model_name)
            except ImportError:
                logger.warning(
                    "google-generativeai not installed — falling back to template engine. "
                    "Install with: pip install google-generativeai"
                )
                self._gemini_key = None
        else:
            logger.info("RiskSummarizer: No GEMINI_API_KEY — using template engine")

    @property
    def backend(self) -> str:
        return "gemini" if self._client else "template"

    async def generate(
        self,
        user_id: str,
        risk_score: float,
        top_contributors: list[dict],
        user_features: dict[str, float],
    ) -> str:
        """Generate a risk summary for an anomalous user."""

        if self._client:
            return await self._generate_gemini(
                user_id, risk_score, top_contributors, user_features
            )
        return _generate_template_summary(
            user_id, risk_score, top_contributors, user_features
        )

    async def _generate_gemini(
        self,
        user_id: str,
        risk_score: float,
        top_contributors: list[dict],
        user_features: dict[str, float],
    ) -> str:
        """Call Gemini API for risk summary generation."""
        prompt = _build_prompt(user_id, risk_score, top_contributors, user_features)
        try:
            import asyncio
            response = await asyncio.to_thread(
                self._client.generate_content,
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 400,
                },
            )
            return response.text.strip()
        except Exception as e:
            logger.error("Gemini API call failed: %s — falling back to template", e)
            return _generate_template_summary(
                user_id, risk_score, top_contributors, user_features
            )
