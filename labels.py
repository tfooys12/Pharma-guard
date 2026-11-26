"""Ground-truth labels.

Labels come straight from the *known simulation event* (``event_mask`` built by
the scenario). No detection algorithm, threshold check or statistics are used,
so these labels can safely be used to evaluate a detector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ScenarioConfig

LABELS: tuple[str, ...] = (
    "NORMAL",
    "TEMPERATURE_SPIKE",
    "PROLONGED_EXCURSION",
    "SENSOR_FAULT",
    "NOISY_SENSOR",
    "UNUSUAL_PATTERN",
    "SHIPMENT_DELAY",
)

# scenario_type (what we simulate) -> ground_truth_label (what the records are)
SCENARIO_TO_LABEL: dict[str, str] = {
    "NORMAL": "NORMAL",
    "SUDDEN_SPIKE": "TEMPERATURE_SPIKE",
    "PROLONGED_EXCURSION": "PROLONGED_EXCURSION",
    "SENSOR_FAULT": "SENSOR_FAULT",
    "NOISY_SENSOR": "NOISY_SENSOR",
    "UNUSUAL_PATTERN": "UNUSUAL_PATTERN",
    "SHIPMENT_DELAY": "SHIPMENT_DELAY",
}

# Columns a detector must NOT see when it is being evaluated.
GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "scenario_type",
    "ground_truth_label",
    "ground_truth_risk",
    "event_id",
    "true_temperature",
)


def apply_labels(df: pd.DataFrame, event_mask: np.ndarray, cfg: ScenarioConfig) -> pd.DataFrame:
    """Add scenario_type, ground_truth_label, ground_truth_risk and event_id.

    Records inside the injected event get the scenario's label, risk 1 and the
    event_id. All other records are NORMAL, risk 0, event_id None.
    """
    event_id = cfg.event_id or f"{cfg.shipment_id}-EVT01"
    label = SCENARIO_TO_LABEL[cfg.scenario_type]
    df["scenario_type"] = cfg.scenario_type
    df["ground_truth_label"] = np.where(event_mask, label, "NORMAL")
    df["ground_truth_risk"] = event_mask.astype(int)
    df["event_id"] = np.where(event_mask, event_id, None)
    return df


def strip_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Return the data a detector is allowed to see (no labels, no true temperature)."""
    return df.drop(columns=[c for c in GROUND_TRUTH_COLUMNS if c in df.columns])
