"""Generate many labeled shipments in one call."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import FAULT_TYPES, PATTERN_TYPES, SCENARIO_TYPES, ScenarioConfig
from .generator import generate_scenario


def generate_batch(
    scenarios: Iterable[str] | None = None,
    samples_per_scenario: int = 10,
    seed: int = 42,
    base_config: ScenarioConfig | None = None,
    randomize: bool = True,
) -> pd.DataFrame:
    """Generate ``samples_per_scenario`` shipments for each scenario type.

    Args:
        scenarios: scenario names (default: all seven).
        samples_per_scenario: number of shipments per scenario.
        seed: master seed. Same seed -> identical DataFrame; new seed -> new data.
        base_config: shared settings (target, thresholds, duration, ...).
        randomize: if True, each shipment gets slightly different noise level,
            starting temperature, severity and direction, and SENSOR_FAULT /
            UNUSUAL_PATTERN shipments cycle through all fault / pattern types.
            If False, only the seeds and IDs differ.

    Every shipment has a unique shipment_id, batch_id and (for event records) event_id.
    """
    names = [s.upper() for s in (scenarios if scenarios is not None else SCENARIO_TYPES)]
    unknown = [s for s in names if s not in SCENARIO_TYPES]
    if unknown:
        raise ValueError(f"Unknown scenarios {unknown}. Choose from {SCENARIO_TYPES}.")
    if samples_per_scenario < 1 or not names:
        raise ValueError("Need at least one scenario and samples_per_scenario >= 1.")

    base = base_config if base_config is not None else ScenarioConfig()
    seeds = np.random.SeedSequence(seed).generate_state(len(names) * samples_per_scenario)

    frames = []
    for s_idx, scenario in enumerate(names):
        for i in range(samples_per_scenario):
            k = s_idx * samples_per_scenario + i
            child = int(seeds[k])
            cfg = base.with_updates(
                shipment_id=f"SHIP{k + 1:04d}",
                batch_id=f"BATCH{k + 1:04d}",
                scenario_type=scenario,
                random_seed=child,
                event_id=None,
            )
            if randomize:
                prng = np.random.default_rng([child, 1])  # separate stream for parameters
                changes = dict(
                    noise_level=round(float(prng.uniform(0.08, 0.25)), 3),
                    severity=round(base.severity * float(prng.uniform(0.8, 1.5)), 2),
                    direction="up" if prng.random() < 0.5 else "down",
                    fault_type=FAULT_TYPES[i % len(FAULT_TYPES)],
                    pattern_type=PATTERN_TYPES[i % len(PATTERN_TYPES)],
                )
                if base.starting_temperature is None:
                    changes["starting_temperature"] = round(
                        base.target_temperature + float(prng.normal(0.0, 0.3)), 2)
                cfg = cfg.with_updates(**changes)
            frames.append(generate_scenario(cfg))

    return pd.concat(frames, ignore_index=True)
