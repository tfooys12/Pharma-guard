"""Base cold-chain time-series generator and the main ``generate_scenario`` API.

Real (hidden) temperature model, per sample t:

    temp(t) = temp(t-1)
              + thermal_response * (setpoint(t) - temp(t-1))   # pull toward setpoint
              + natural_variation                              # small random change
    setpoint(t) = target + slow compressor cycle + scenario effect(t)

What the sensor reports is ``temp(t) + sensor noise``, then any sensor fault.
Because each value builds on the previous one, scenario effects rise and
recover smoothly instead of jumping between unrelated values.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from . import labels, noise, scenarios
from .config import ScenarioConfig

OUTPUT_COLUMNS: list[str] = [
    "timestamp", "shipment_id", "batch_id", "temperature",
    "target_temperature", "lower_threshold", "upper_threshold", "sensor_status",
    "scenario_type", "ground_truth_label", "ground_truth_risk", "event_id",
    # extras
    "true_temperature", "elapsed_minutes",
    "expected_duration_minutes", "actual_duration_minutes",
]


def simulate_true_temperature(
    cfg: ScenarioConfig,
    rng: np.random.Generator,
    n: int,
    effect: np.ndarray,
    response: float | None = None,
) -> np.ndarray:
    """Simulate the real product temperature (no sensor noise) for ``n`` samples."""
    k = cfg.thermal_response if response is None else response
    start = cfg.starting_temperature
    if start is None:
        start = cfg.target_temperature + rng.normal(0.0, 0.2)

    minutes = np.arange(n) * cfg.sampling_interval_minutes
    cycle_period = rng.uniform(20.0, 40.0)  # compressor on/off cycle, minutes
    phase = rng.uniform(0.0, 2 * np.pi)
    cycle = cfg.cycle_amplitude * np.sin(2 * np.pi * minutes / cycle_period + phase)
    variation = rng.normal(0.0, cfg.natural_variation, n)
    setpoint = cfg.target_temperature + cycle + effect

    temp = np.empty(n)
    prev = start
    for t in range(n):
        prev = prev + k * (setpoint[t] - prev) + variation[t]
        temp[t] = prev
    return temp


def generate_scenario(
    config: ScenarioConfig | None = None, seed: int | None = None, **overrides
) -> pd.DataFrame:
    """Generate one labeled shipment as a DataFrame.

    Args:
        config: a :class:`ScenarioConfig`. Defaults to ``ScenarioConfig()``.
        seed: shortcut for ``random_seed`` (``generate_scenario(seed=42)``).
        **overrides: any other ScenarioConfig field, e.g. ``scenario_type="SUDDEN_SPIKE"``.

    The same config and seed always give identical output.
    """
    cfg = config if config is not None else ScenarioConfig()
    changes = dict(overrides)
    if seed is not None:
        changes["random_seed"] = seed
    if changes:
        cfg = cfg.with_updates(**changes)

    rng = np.random.default_rng(cfg.random_seed)
    n = cfg.n_samples

    plan = scenarios.build_plan(cfg, rng, n)
    true_temp = simulate_true_temperature(cfg, rng, n, plan.effect, plan.response)

    measured = true_temp + noise.gaussian_noise(rng, n, cfg.noise_level)
    status = np.full(n, "OK", dtype=object)
    for op in plan.sensor_ops:
        measured, status = op(measured, status)

    elapsed = np.arange(n) * cfg.sampling_interval_minutes
    expected = plan.expected_duration_minutes
    df = pd.DataFrame({
        "timestamp": pd.date_range(cfg.start_time, periods=n,
                                   freq=pd.Timedelta(minutes=cfg.sampling_interval_minutes)),
        "shipment_id": cfg.shipment_id,
        "batch_id": cfg.batch_id,
        "temperature": np.round(measured, 2),
        "target_temperature": cfg.target_temperature,
        "lower_threshold": cfg.lower_threshold,
        "upper_threshold": cfg.upper_threshold,
        "sensor_status": status,
    })
    df = labels.apply_labels(df, plan.event_mask, cfg)
    df["true_temperature"] = np.round(true_temp, 2)
    df["elapsed_minutes"] = elapsed
    df["expected_duration_minutes"] = cfg.duration_minutes if expected is None else expected
    df["actual_duration_minutes"] = cfg.duration_minutes

    df = df[OUTPUT_COLUMNS]
    df.attrs["config"] = asdict(cfg)
    return df
