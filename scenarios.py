"""Scenario definitions.

Each ``build_*`` function turns a :class:`ScenarioConfig` into a
:class:`ScenarioPlan`, which describes the *known* simulation event:

* ``effect``      - smooth offset (deg C) applied to the real temperature setpoint
* ``response``    - optional override of how fast the real temperature follows it
* ``sensor_ops``  - changes applied to what the sensor *reports* (faults, noise)
* ``event_mask``  - which records belong to the injected event (the ground truth).
                    For temperature effects it includes the short recovery tail
                    while the real temperature settles back to normal.

The generator turns the plan into data; ``labels.py`` turns ``event_mask`` into
labels. No detection logic is used anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import noise
from .config import FAULT_TYPES, PATTERN_TYPES, ScenarioConfig

# A sensor op receives (values, status) and returns modified copies.
SensorOp = Callable[[np.ndarray, np.ndarray], "tuple[np.ndarray, np.ndarray]"]


@dataclass
class ScenarioPlan:
    event_mask: np.ndarray
    effect: np.ndarray
    response: float | None = None
    sensor_ops: list[SensorOp] = field(default_factory=list)
    expected_duration_minutes: float | None = None
    details: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _samples(minutes: float, cfg: ScenarioConfig) -> int:
    return max(1, int(round(minutes / cfg.sampling_interval_minutes)))


def _direction(cfg: ScenarioConfig, rng: np.random.Generator) -> int:
    if cfg.direction == "random":
        return 1 if rng.random() < 0.5 else -1
    return 1 if cfg.direction == "up" else -1


def _choose(value: str, options: tuple[str, ...], rng: np.random.Generator) -> str:
    return str(rng.choice(options)) if value == "RANDOM" else value


def _margin(cfg: ScenarioConfig, sign: int) -> float:
    """Distance from target to the safe-range limit in the given direction."""
    if sign > 0:
        return cfg.upper_threshold - cfg.target_temperature
    return cfg.target_temperature - cfg.lower_threshold


def _pick_window(
    cfg: ScenarioConfig, rng: np.random.Generator, n: int, default_minutes: tuple[float, float]
) -> tuple[int, int]:
    """Return (start, end) sample indices of the event. Uses config values if set."""
    if cfg.event_duration_minutes is not None:
        length = _samples(cfg.event_duration_minutes, cfg)
    else:
        length = _samples(rng.uniform(*default_minutes), cfg)
    length = max(1, min(length, n - 1))

    if cfg.event_start_minute is not None:
        start = int(round(cfg.event_start_minute / cfg.sampling_interval_minutes))
        if not 0 <= start < n:
            raise ValueError("event_start_minute is outside the shipment duration.")
    else:
        earliest = int(0.15 * n)
        latest = max(earliest, n - length - int(0.10 * n))
        start = int(rng.integers(earliest, latest + 1))
    return start, min(n, start + length)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _envelope(n: int, start: int, end: int, ramp_up: int, ramp_down: int) -> np.ndarray:
    """0 outside [start, end); smooth rise, plateau of 1, smooth fall inside."""
    idx = np.arange(n)
    up = _smoothstep((idx - start + 1) / max(ramp_up, 1))
    down = _smoothstep((end - idx) / max(ramp_down, 1))
    return np.where((idx >= start) & (idx < end), np.minimum(up, down), 0.0)


def _with_recovery(mask: np.ndarray, response: float) -> np.ndarray:
    """Extend the event window by the time the real temperature needs to recover.

    The real temperature follows its setpoint with inertia, so it is still
    deviating for a few samples after the injected effect has ended. About
    3/response samples bring it back to within ~5% of normal.
    """
    out = mask.copy()
    last = int(np.flatnonzero(mask).max())
    tail = int(np.ceil(3.0 / response))
    out[last + 1: last + 1 + tail] = True
    return out


def _empty(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(n, dtype=bool), np.zeros(n)


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
def build_normal(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """No event: temperature hovers around the target."""
    mask, effect = _empty(n)
    return ScenarioPlan(mask, effect)


def build_sudden_spike(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """Short, sharp excursion (e.g. door left open) that rises and recovers smoothly."""
    sign = _direction(cfg, rng)
    start, end = _pick_window(cfg, rng, n, (4, 10))
    length = end - start
    magnitude = cfg.magnitude if cfg.magnitude is not None else _margin(cfg, sign) + 4.0 * cfg.severity
    env = _envelope(n, start, end, max(1, length // 4), max(1, length // 3))
    mask, _ = _empty(n)
    mask[start:end] = True
    mask = _with_recovery(mask, 0.6)
    # high response -> the real temperature follows quickly, but not instantly
    return ScenarioPlan(mask, sign * magnitude * env, response=0.6,
                        details={"direction": sign, "magnitude": magnitude})


def build_prolonged_excursion(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """Temperature held outside the safe range for a long time (e.g. failed cooling)."""
    sign = _direction(cfg, rng)
    total = cfg.duration_minutes
    start, end = _pick_window(cfg, rng, n, (0.25 * total, 0.40 * total))
    length = end - start
    magnitude = cfg.magnitude if cfg.magnitude is not None else _margin(cfg, sign) + 1.0 * cfg.severity
    ramp = max(2, length // 8)
    env = _envelope(n, start, end, ramp, ramp)
    mask, _ = _empty(n)
    mask[start:end] = True
    mask = _with_recovery(mask, cfg.thermal_response)
    return ScenarioPlan(mask, sign * magnitude * env,
                        details={"direction": sign, "magnitude": magnitude})


def build_sensor_fault(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """Real temperature is fine; the *sensor* misreports. Type set by ``fault_type``."""
    fault = _choose(cfg.fault_type, FAULT_TYPES, rng)
    sign = _direction(cfg, rng)
    total = cfg.duration_minutes
    start, end = _pick_window(cfg, rng, n, (0.10 * total, 0.25 * total))
    length = end - start
    mask, effect = _empty(n)
    mask[start:end] = True
    ops: list[SensorOp] = []

    if fault == "FROZEN":
        ops.append(lambda v, s: (noise.frozen_values(v, start, end), s))

    elif fault == "OFFLINE":
        def offline(v, s):
            v, s = v.copy(), s.copy()
            v[start:end] = np.nan
            s[start:end] = "OFFLINE"
            return v, s
        ops.append(offline)

    elif fault == "MISSING":
        drop = noise.missing_mask(rng, n, 0.35, mask)
        if not drop.any():
            drop[start + length // 2] = True
        mask = drop  # only the lost readings are faulty records

        def missing(v, s):
            v, s = v.copy(), s.copy()
            v[drop] = np.nan
            s[drop] = "NO_DATA"
            return v, s
        ops.append(missing)

    elif fault == "UNREALISTIC":
        hit = noise.missing_mask(rng, n, 0.20, mask)
        if not hit.any():
            hit[start + length // 2] = True
        mask = hit  # only the impossible readings are faulty records
        bad = rng.choice([85.0, -127.0, 125.0], n)  # typical sensor error codes

        def unrealistic(v, s):
            v = v.copy()
            v[hit] = bad[hit]
            return v, s
        ops.append(unrealistic)

    elif fault == "JUMP":
        offset = sign * (cfg.magnitude if cfg.magnitude is not None else 4.0 * cfg.severity)
        ops.append(lambda v, s: (v + noise.sensor_jump(n, start, end, offset), s))

    elif fault == "DRIFT":
        total_drift = sign * (cfg.magnitude if cfg.magnitude is not None else 3.0 * cfg.severity)
        rate = total_drift / length
        ops.append(lambda v, s: (v + noise.sensor_drift(n, start, end, rate), s))

    # Faults are "silent" except lost data, as on a real device: the reported
    # status is only OK / NO_DATA / OFFLINE.
    return ScenarioPlan(mask, effect, sensor_ops=ops, details={"fault_type": fault})


def build_noisy_sensor(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """Real temperature stays stable; readings become very noisy (zero-mean, no shift)."""
    total = cfg.duration_minutes
    start, end = _pick_window(cfg, rng, n, (0.20 * total, 0.50 * total))
    mask, effect = _empty(n)
    mask[start:end] = True
    std = cfg.noise_level * 8.0 * cfg.severity
    extra = noise.high_noise(rng, n, std, mask)
    outliers = noise.random_spikes(rng, n, 0.03, 3 * std, mask)
    return ScenarioPlan(mask, effect, sensor_ops=[lambda v, s: (v + extra + outliers, s)],
                        details={"noise_std": std})


def build_unusual_pattern(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """Abnormal shapes that mostly stay inside the safe range (severity > ~1.6 may cross it)."""
    pattern = _choose(cfg.pattern_type, PATTERN_TYPES, rng)
    sign = _direction(cfg, rng)
    total = cfg.duration_minutes
    start, end = _pick_window(cfg, rng, n, (0.30 * total, 0.60 * total))
    length = end - start
    idx = np.arange(n)
    amp = cfg.magnitude if cfg.magnitude is not None else 0.6 * _margin(cfg, sign) * cfg.severity
    taper = _envelope(n, start, end, max(1, length // 10), max(1, length // 10))
    mask, _ = _empty(n)
    mask[start:end] = True
    response: float | None = 0.6

    if pattern == "DRIFT":
        ramp = np.where((idx >= start) & (idx < end), (idx - start + 1) / length, 0.0)
        effect = sign * amp * ramp * _envelope(n, start, end, 1, max(2, length // 10))
        response = None  # slow thermal response suits a gradual drift
    elif pattern == "REPEATED_SPIKES":
        k = int(rng.integers(3, 7))
        width = max(2, _samples(2, cfg))
        peaks = np.zeros(n)
        for c in np.linspace(start, max(start, end - width), k).astype(int):
            peaks = np.maximum(peaks, _envelope(n, c, min(n, c + width), 1, max(1, width // 2)))
        effect = sign * amp * peaks
    elif pattern == "OSCILLATION":
        period = max(rng.uniform(8, 20), 4 * cfg.sampling_interval_minutes)
        phase = 2 * np.pi * (idx - start) * cfg.sampling_interval_minutes / period
        effect = amp * np.sin(phase) * taper
    else:  # RAPID_FLUCTUATION
        effect = rng.uniform(-1, 1, n) * amp * 1.2 * taper
        response = 0.7

    mask = _with_recovery(mask, response if response is not None else cfg.thermal_response)
    return ScenarioPlan(mask, effect, response=response, details={"pattern_type": pattern})


def build_shipment_delay(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """Shipment arrives later than planned. Records after the planned arrival are the event."""
    total = cfg.duration_minutes
    if cfg.expected_duration_minutes is not None:
        expected = float(cfg.expected_duration_minutes)
    else:
        delay = cfg.event_duration_minutes if cfg.event_duration_minutes is not None \
            else rng.uniform(0.25, 0.40) * total
        expected = total - delay
    if not 0 < expected < total:
        raise ValueError("expected duration must be > 0 and shorter than duration_minutes (the actual time).")

    start = min(int(np.ceil(expected / cfg.sampling_interval_minutes)), n - 1)
    idx = np.arange(n)
    mask, effect = _empty(n)
    mask[start:] = True

    if cfg.delay_deterioration:
        magnitude = cfg.magnitude if cfg.magnitude is not None \
            else 0.8 * _margin(cfg, 1) * cfg.severity
        ramp = np.where(idx >= start, (idx - start + 1) / max(n - start, 1), 0.0)
        effect = magnitude * ramp

    return ScenarioPlan(mask, effect, expected_duration_minutes=expected,
                        details={"delay_minutes": total - expected})


SCENARIO_BUILDERS = {
    "NORMAL": build_normal,
    "SUDDEN_SPIKE": build_sudden_spike,
    "PROLONGED_EXCURSION": build_prolonged_excursion,
    "SENSOR_FAULT": build_sensor_fault,
    "NOISY_SENSOR": build_noisy_sensor,
    "UNUSUAL_PATTERN": build_unusual_pattern,
    "SHIPMENT_DELAY": build_shipment_delay,
}


def build_plan(cfg: ScenarioConfig, rng: np.random.Generator, n: int) -> ScenarioPlan:
    """Dispatch to the builder for ``cfg.scenario_type``."""
    return SCENARIO_BUILDERS[cfg.scenario_type](cfg, rng, n)
