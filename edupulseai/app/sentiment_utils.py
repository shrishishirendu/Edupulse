"""Shared sentiment helpers for app/demo-safe labeling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "happy",
    "motivated",
    "improving",
    "improved",
    "progress",
    "confident",
    "engaged",
    "supportive",
    "helpful",
    "clear",
    "understand",
    "understanding",
    "better",
    "strong",
    "enjoy",
    "satisfied",
    "encouraged",
}

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "sad",
    "stressed",
    "anxious",
    "frustrated",
    "confused",
    "overwhelmed",
    "struggling",
    "struggle",
    "difficult",
    "hard",
    "missed",
    "behind",
    "low",
    "worried",
    "disappointed",
    "tired",
    "drop",
    "dropping",
}

LABEL_MAP = {
    "positive": "🟢 Positive",
    "neutral": "🟡 Neutral",
    "negative": "🔴 Negative",
    "🟢 positive": "🟢 Positive",
    "🟡 neutral": "🟡 Neutral",
    "🔴 negative": "🔴 Negative",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_student_id(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    if text.isdigit():
        trimmed = text.lstrip("0")
        return trimmed if trimmed else "0"
    return text


def _map_existing_label(value: Any) -> str | None:
    key = _safe_text(value).lower()
    if not key:
        return None
    return LABEL_MAP.get(key)


def _hash_bucket_label(student_id: Any) -> str:
    sid = _normalize_student_id(student_id) or "missing"
    bucket = int(hashlib.md5(sid.encode("utf-8")).hexdigest()[:2], 16)
    if bucket <= 84:
        return "🔴 Negative"
    if bucket <= 169:
        return "🟡 Neutral"
    return "🟢 Positive"


def infer_sentiment_label(feedback_text: str) -> str:
    text = _safe_text(feedback_text).lower()
    if not text:
        return "🟡 Neutral"
    words = [token.strip(".,!?;:\"'()[]{}") for token in text.split()]
    pos_count = sum(1 for token in words if token in POSITIVE_WORDS)
    neg_count = sum(1 for token in words if token in NEGATIVE_WORDS)
    score = pos_count - neg_count
    if score >= 1:
        return "🟢 Positive"
    if score <= -1:
        return "🔴 Negative"
    return "🟡 Neutral"


def load_latest_feedback(feedback_csv_path: Path | str) -> pd.DataFrame:
    path = Path(feedback_csv_path)
    if not path.exists():
        return pd.DataFrame(columns=["student_id", "latest_feedback"])

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["student_id", "latest_feedback"])

    if df.empty:
        return pd.DataFrame(columns=["student_id", "latest_feedback"])

    df.columns = [str(col).strip() for col in df.columns]
    id_candidates = ["student_id", "studentid", "id"]
    text_candidates = ["text", "feedback", "review_text", "comment", "latest_feedback"]
    time_candidates = ["timestamp", "time", "created_at", "date", "datetime"]

    id_col = next((col for col in id_candidates if col in df.columns), None)
    text_col = next((col for col in text_candidates if col in df.columns), None)
    time_col = next((col for col in time_candidates if col in df.columns), None)

    if id_col is None or text_col is None:
        return pd.DataFrame(columns=["student_id", "latest_feedback"])

    small = df[[id_col, text_col] + ([time_col] if time_col else [])].copy()
    small = small.rename(columns={id_col: "student_id", text_col: "latest_feedback"})
    small["student_id"] = small["student_id"].map(_normalize_student_id)
    small["latest_feedback"] = small["latest_feedback"].map(_safe_text)
    small = small[(small["student_id"] != "") & (small["latest_feedback"] != "")]
    if small.empty:
        return pd.DataFrame(columns=["student_id", "latest_feedback"])

    if time_col:
        small["_parsed_ts"] = pd.to_datetime(small[time_col], errors="coerce")
        small = small.sort_values(by=["student_id", "_parsed_ts"], kind="stable")
    else:
        small = small.sort_values(by=["student_id"], kind="stable")

    latest = small.groupby("student_id", as_index=False).tail(1)
    return latest[["student_id", "latest_feedback"]].reset_index(drop=True)


def ensure_sentiment_labels(
    df: pd.DataFrame, student_id_col: str = "student_id", feedback_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame() if df is None else df.copy()
        if "sentiment_label" not in out.columns:
            out["sentiment_label"] = pd.Series(dtype="object")
        return out

    out = df.copy()
    if "sentiment_label" not in out.columns:
        out["sentiment_label"] = ""

    out["_existing_mapped"] = out["sentiment_label"].map(_map_existing_label)
    out["_has_feedback"] = False
    out["_feedback_label"] = None

    if feedback_df is not None and not feedback_df.empty and student_id_col in out.columns and "student_id" in feedback_df.columns:
        fdf = feedback_df.copy()
        fdf["student_id"] = fdf["student_id"].map(_normalize_student_id)
        if "latest_feedback" not in fdf.columns:
            fdf["latest_feedback"] = ""
        fdf["latest_feedback"] = fdf["latest_feedback"].map(_safe_text)
        out[student_id_col] = out[student_id_col].map(_normalize_student_id)
        out = out.merge(
            fdf[["student_id", "latest_feedback"]],
            left_on=student_id_col,
            right_on="student_id",
            how="left",
            suffixes=("", "_fb"),
        )
        out["_has_feedback"] = out["latest_feedback"].map(_safe_text) != ""
        out.loc[out["_has_feedback"], "_feedback_label"] = out.loc[out["_has_feedback"], "latest_feedback"].map(
            infer_sentiment_label
        )
    elif student_id_col in out.columns:
        out[student_id_col] = out[student_id_col].map(_normalize_student_id)
        out["latest_feedback"] = ""
    else:
        out["latest_feedback"] = ""

    out["sentiment_label"] = out["_existing_mapped"]
    out.loc[out["_has_feedback"], "sentiment_label"] = out.loc[out["_has_feedback"], "_feedback_label"]

    missing_mask = out["sentiment_label"].isna() | (out["sentiment_label"].map(_safe_text) == "")
    if student_id_col in out.columns:
        out.loc[missing_mask, "sentiment_label"] = out.loc[missing_mask, student_id_col].map(_hash_bucket_label)
    else:
        out.loc[missing_mask, "sentiment_label"] = "🟡 Neutral"

    for col in ["student_id_fb", "_existing_mapped", "_has_feedback", "_feedback_label"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    return out
