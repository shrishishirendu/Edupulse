"""Intervention playbook utilities for dropout risk mitigation."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

URGENCY_SCORES = {"low": 0, "medium": 1, "high": 2}


def load_playbook(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the interventions playbook from configuration."""
    interventions_cfg = cfg.get("interventions") or {}
    playbook = interventions_cfg.get("playbook") or {}
    if not isinstance(playbook, dict):
        raise ValueError("interventions.playbook must be a mapping.")
    return playbook


def _match_column(feature: str, columns: List[str]) -> str | None:
    """Find the best matching column for a feature name."""
    base = feature.split("=", 1)[0].strip()
    candidates = [base]
    if "__" in base:
        candidates.append(base.split("__", 1)[1].strip())

    lower_columns = {col.lower().strip(): col for col in columns}

    for cand in candidates:
        lc = cand.lower().strip()
        if lc in lower_columns:
            return lower_columns[lc]
        for col in columns:
            col_lower = col.lower().strip()
            if lc in col_lower or col_lower in lc:
                return col
    return None


def _base_feature_name(feature: str) -> str:
    """Extract a human-readable base feature name from transformer output."""
    base = str(feature)
    if "=" in base:
        base = base.split("=", 1)[0]
    if "__" in base:
        base = base.split("__", 1)[1]
    return base.strip()


def highest_urgency(urgencies: List[str], default: str) -> str:
    """Return the most severe urgency from a list."""
    best = default
    best_score = URGENCY_SCORES.get(str(default).lower(), -1)
    for urgency in urgencies:
        score = URGENCY_SCORES.get(str(urgency).lower(), -1)
        if score > best_score:
            best = urgency
            best_score = score
    return best


def infer_top_reasons(row: pd.Series, top_drivers_df: pd.DataFrame, max_reasons: int = 3) -> List[str]:
    """Infer top reasons based on active driver features present in the row."""
    if top_drivers_df is None or top_drivers_df.empty:
        return []

    reasons: List[str] = []
    columns = list(row.index)

    for feature in top_drivers_df["feature"]:
        base_name = _base_feature_name(str(feature))
        matched_col = _match_column(base_name, columns)
        if matched_col is None:
            continue

        value = row.get(matched_col)
        if pd.isna(value):
            continue

        is_active = True
        if isinstance(value, str):
            is_active = value.strip() != ""
        else:
            try:
                is_active = float(value) != 0.0
            except Exception:
                is_active = True

        if not is_active:
            continue

        reason_label = base_name or matched_col
        if reason_label not in reasons:
            reasons.append(reason_label)
        if len(reasons) >= max_reasons:
            break

    return reasons


def recommend_actions(top_reasons: List[str], playbook: Dict[str, Any]) -> Dict[str, Any]:
    """Recommend interventions based on matched playbook rules."""
    rules = playbook.get("rules") or []
    default_actions = playbook.get("default_actions") or []
    default_owner = playbook.get("default_owner", "General")
    default_urgency = playbook.get("default_urgency", "low")

    matched_actions: List[str] = []
    matched_rules: List[str] = []
    owners: List[str] = []
    matched_urgencies: List[str] = []

    for idx, rule in enumerate(rules):
        tokens = [str(tok).lower().strip() for tok in (rule.get("when_any_features_contain") or [])]
        tokens = [tok for tok in tokens if tok]
        if not tokens:
            continue
        reasons_normalized = [str(reason).lower().strip() for reason in top_reasons]
        if any(tok in reason for tok in tokens for reason in reasons_normalized):
            matched_rules.append(rule.get("name", f"rule_{idx}"))

            for action in rule.get("actions") or []:
                if action not in matched_actions:
                    matched_actions.append(action)
            owners.append(rule.get("owner", default_owner))
            matched_urgencies.append(rule.get("urgency", default_urgency))

    if not matched_rules:
        matched_actions = default_actions
        owner = default_owner
        urgency = default_urgency
    else:
        distinct_owners = {own for own in owners if own}
        if len(distinct_owners) > 1:
            owner = "Multiple"
        elif len(distinct_owners) == 1:
            owner = distinct_owners.pop()
        else:
            owner = default_owner

        urgency = highest_urgency(matched_urgencies, default_urgency)

    return {
        "actions": matched_actions,
        "owner": owner,
        "urgency": urgency,
        "matched_rules": matched_rules,
    }


def compute_priority(risk_band: str, urgency: str, cfg: Dict[str, Any]) -> int:
    """Compute priority score based on risk band and urgency weights (lower is higher priority)."""
    interventions_cfg = cfg.get("interventions") or {}
    risk_priority = interventions_cfg.get("risk_band_priority") or {}
    risk_score = int(risk_priority.get(str(risk_band), 100))

    urgency_weights = {"high": 0, "medium": 1, "low": 2}
    urgency_score = urgency_weights.get(str(urgency).lower(), 3)

    return risk_score * 10 + urgency_score
