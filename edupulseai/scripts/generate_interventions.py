"""Generate intervention recommendations based on dropout predictions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from edupulse.interventions.playbook import compute_priority, infer_top_reasons, load_playbook, recommend_actions

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"
URGENCY_ORDER = {"low": 0, "medium": 1, "high": 2}


def max_urgency(a: str, b: str) -> str:
    """Return the more severe urgency level."""
    a_score = URGENCY_ORDER.get(str(a).lower(), -1)
    b_score = URGENCY_ORDER.get(str(b).lower(), -1)
    return a if a_score >= b_score else b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate intervention recommendations for dropout predictions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML (default: configs/config.yaml).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("Missing dependency: PyYAML. Add PyYAML to requirements.txt and redeploy.")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a mapping.")
    return cfg


def resolve_config_path(config_path: Path) -> Path:
    cfg_path = config_path
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return cfg_path


def _load_predictions() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reports_dir = PROJECT_ROOT / "reports"
    val_pred_path = reports_dir / "dropout_val_predictions.csv"
    test_pred_path = reports_dir / "dropout_test_predictions.csv"
    top_drivers_path = reports_dir / "dropout_top_drivers.csv"

    for path in (val_pred_path, test_pred_path, top_drivers_path):
        if not path.exists():
            raise FileNotFoundError(f"Required report not found: {path}")

    val_preds = pd.read_csv(val_pred_path)
    test_preds = pd.read_csv(test_pred_path)
    top_drivers = pd.read_csv(top_drivers_path)
    return val_preds, test_preds, top_drivers


def _load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"
    for path in (val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Processed split not found: {path}")
    return pd.read_csv(val_path), pd.read_csv(test_path)


def _merge_predictions(
    split_df: pd.DataFrame, preds_df: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    dataset_cfg = cfg.get("dataset") or {}
    id_column = dataset_cfg.get("id_column")
    target_column = dataset_cfg.get("target_column")

    if id_column and id_column in split_df.columns and id_column in preds_df.columns:
        merged = split_df.merge(preds_df, on=id_column, how="left", suffixes=("", "_pred"))
    else:
        left = split_df.drop(columns=[target_column], errors="ignore").reset_index(drop=True)
        right = preds_df.reset_index(drop=True)
        merged = pd.concat([left, right], axis=1)

    if id_column and id_column not in merged.columns and id_column in split_df.columns:
        merged[id_column] = split_df[id_column].values

    return merged


def _create_interventions(
    merged_df: pd.DataFrame, top_drivers: pd.DataFrame, cfg: Dict[str, Any]
) -> pd.DataFrame:
    playbook = load_playbook(cfg)
    dataset_cfg = cfg.get("dataset") or {}
    id_column = dataset_cfg.get("id_column")

    records = []
    for _, row in merged_df.iterrows():
        top_reasons = infer_top_reasons(row, top_drivers)
        rec = recommend_actions(top_reasons, playbook)
        risk_band = str(row.get("risk_band") or "")
        risk_band_lower = risk_band.lower()

        band_urgency = {"high": "high", "medium": "medium", "low": "low"}.get(risk_band_lower, rec.get("urgency"))
        rule_urgency = rec.get("urgency") or band_urgency
        final_urgency = max_urgency(rule_urgency, band_urgency)
        priority = compute_priority(risk_band_lower, final_urgency, cfg)

        record = {
            "dropout_risk": row.get("dropout_risk"),
            "risk_band": row.get("risk_band"),
            "y_pred": row.get("y_pred"),
            "y_true": row.get("y_true"),
            "top_reasons": "; ".join(top_reasons),
            "recommended_actions": "; ".join(rec.get("actions") or []),
            "owner": rec.get("owner"),
            "urgency": final_urgency,
            "priority": priority,
            "matched_rules": "; ".join(rec.get("matched_rules") or []),
        }
        if id_column and id_column in row:
            record[id_column] = row[id_column]

        records.append(record)

    columns = []
    if id_column:
        columns.append(id_column)
    columns.extend(
        [
            "dropout_risk",
            "risk_band",
            "y_pred",
            "y_true",
            "top_reasons",
            "recommended_actions",
            "owner",
            "urgency",
            "priority",
            "matched_rules",
        ]
    )
    df = pd.DataFrame(records, columns=columns)

    risk_lower = df["risk_band"].astype(str).str.lower()
    urgency_lower = df["urgency"].astype(str).str.lower()

    high_violations = ((risk_lower == "high") & (urgency_lower != "high")).sum()

    if high_violations:
        raise ValueError(f"Urgency floor violated for high risk rows: {high_violations} rows not high urgency.")

    return df


def _print_summary(df: pd.DataFrame) -> None:
    import pandas as pd

    assert not ((df["risk_band"] == "high") & (df["urgency"] != "high")).any()

    band_counts = df["risk_band"].value_counts()
    urgency_counts = df["urgency"].value_counts()

    assert not ((df["risk_band"] == "high") & (df["urgency"] != "high")).any()
    assert not ((df["risk_band"] == "medium") & (df["urgency"] == "low")).any()

    actions_counter: Counter[str] = Counter()
    matched_rules_counter: Counter[str] = Counter()
    default_count = 0

    for actions in df["recommended_actions"]:
        if not isinstance(actions, str):
            continue
        for action in [part.strip() for part in actions.split(";") if part.strip()]:
            actions_counter[action] += 1

    for matched in df["matched_rules"]:
        if not isinstance(matched, str) or matched.strip() == "":
            default_count += 1
            continue
        for rule in [part.strip() for part in matched.split(";") if part.strip()]:
            matched_rules_counter[rule] += 1

    total_rows = len(df) or 1
    default_pct = (default_count / total_rows) * 100

    print("Intervention summary (test set):")
    print(f"Risk bands: {band_counts.to_dict()}")
    print(f"Urgency counts: {urgency_counts.to_dict()}")
    print("Urgency by risk band (crosstab):")
    print(pd.crosstab(df["risk_band"], df["urgency"]).to_dict())
    top_actions = actions_counter.most_common(5)
    print(f"Top actions: {top_actions}")
    top_rules = matched_rules_counter.most_common(5)
    print(f"Matched rules: {top_rules}")
    print(f"Default actions used in {default_pct:.1f}% of rows")


def main() -> None:
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    cfg = load_config(cfg_path)

    val_preds, test_preds, top_drivers = _load_predictions()
    val_split, test_split = _load_splits()

    val_merged = _merge_predictions(val_split, val_preds, cfg)
    test_merged = _merge_predictions(test_split, test_preds, cfg)

    val_interventions = _create_interventions(val_merged, top_drivers, cfg)
    test_interventions = _create_interventions(test_merged, top_drivers, cfg)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    val_interventions.to_csv(reports_dir / "dropout_interventions_val.csv", index=False)
    test_interventions.to_csv(reports_dir / "dropout_interventions_test.csv", index=False)

    _print_summary(test_interventions)


if __name__ == "__main__":
    main()
