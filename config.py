"""Configuration for the cold-chain simulator.

A single dataclass, :class:`ScenarioConfig`, describes one simulated shipment.
``SimulationConfig`` is an alias of the same class, so both names work.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

SCENARIO_TYPES: tuple[str, ...] = (
    "NORMAL",
    "SUDDEN_SPIKE",
    "PROLONGED_EXCURSION",
    "SENSOR_FAULT",
    "NOISY_SENSOR",
    "UNUSUAL_PATTERN",
    "SHIPMENT_DELAY",
)
FAULT_TYPES: tuple[str, ...] = (
    "FROZEN",       # sensor keeps repeating the last good value
    "MISSING",      # individual readings dropped (NaN)
    "UNREALISTIC",  # impossible values such as 85.0 or -127.0
    "JUMP",         # reading is shifted by a constant offset, true temp is fine
    "OFFLINE",      # sensor stops reporting for a while (NaN)
    "DRIFT",        # calibration drift: reading slowly walks away from truth
)
PATTERN_TYPES: tuple[str, ...] = (
    "DRIFT",
    "REPEATED_SPIKES",
    "OSCILLATION",
    "RAPID_FLUCTUATION",
)
DIRECTIONS: tuple[str, ...] = ("up", "down", "random")


@dataclass
class ScenarioConfig:
    """All settings for one simulated shipment. Every field has a default."""

    # --- identity -----------------------------------------------------------
    shipment_id: str = "SHIP001"
    batch_id: str = "BATCH001"
    scenario_type: str = "NORMAL"          # one of SCENARIO_TYPES

    # --- time axis ----------------------------------------------------------
    duration_minutes: float = 180
    sampling_interval_minutes: float = 1
    start_time: datetime = datetime(2025, 1, 1, 8, 0)

    # --- temperature limits (degrees C) ------------------------------------
    target_temperature: float = 5.0
    lower_threshold: float = 2.0
    upper_threshold: float = 8.0
    starting_temperature: float | None = None   # None -> target + small offset

    # --- base dynamics ------------------------------------------------------
    noise_level: float = 0.15          # std-dev of normal sensor noise (deg C)
    natural_variation: float = 0.05    # std-dev of real temperature change per step
    thermal_response: float = 0.3      # 0..1, how fast real temp follows its setpoint
    cycle_amplitude: float = 0.15      # slow compressor-cycle swing (deg C)
    random_seed: int = 42

    # --- event settings (None -> chosen randomly but reproducibly) ----------
    event_start_minute: float | None = None
    event_duration_minutes: float | None = None
    severity: float = 1.0              # multiplies default event magnitudes
    magnitude: float | None = None     # explicit size in deg C (overrides severity default)
    direction: str = "up"              # "up", "down" or "random"
    fault_type: str = "FROZEN"         # SENSOR_FAULT: one of FAULT_TYPES or "RANDOM"
    pattern_type: str = "DRIFT"        # UNUSUAL_PATTERN: one of PATTERN_TYPES or "RANDOM"
    expected_duration_minutes: float | None = None  # SHIPMENT_DELAY: planned transit time
    delay_deterioration: bool = True   # SHIPMENT_DELAY: temperature creeps up while late
    event_id: str | None = None        # None -> "<shipment_id>-EVT01"

    def __post_init__(self) -> None:
        self.scenario_type = self.scenario_type.upper().strip()
        self.direction = self.direction.lower().strip()
        self.fault_type = self.fault_type.upper().strip()
        self.pattern_type = self.pattern_type.upper().strip()

        if self.scenario_type not in SCENARIO_TYPES:
            raise ValueError(f"Unknown scenario_type {self.scenario_type!r}. Choose from {SCENARIO_TYPES}.")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}.")
        if self.fault_type not in FAULT_TYPES + ("RANDOM",):
            raise ValueError(f"fault_type must be one of {FAULT_TYPES} or 'RANDOM'.")
        if self.pattern_type not in PATTERN_TYPES + ("RANDOM",):
            raise ValueError(f"pattern_type must be one of {PATTERN_TYPES} or 'RANDOM'.")
        if self.sampling_interval_minutes <= 0:
            raise ValueError("sampling_interval_minutes must be > 0.")
        if self.n_samples < 10:
            raise ValueError("duration_minutes / sampling_interval_minutes must give at least 10 samples.")
        if not self.lower_threshold < self.target_temperature < self.upper_threshold:
            raise ValueError("Need lower_threshold < target_temperature < upper_threshold.")
        if not 0 < self.thermal_response <= 1:
            raise ValueError("thermal_response must be in (0, 1].")
        if self.severity <= 0:
            raise ValueError("severity must be > 0.")
        if self.noise_level < 0 or self.natural_variation < 0:
            raise ValueError("noise_level and natural_variation must be >= 0.")

    @property
    def n_samples(self) -> int:
        """Number of sensor readings in the simulated shipment."""
        return int(self.duration_minutes // self.sampling_interval_minutes)

    def with_updates(self, **changes) -> "ScenarioConfig":
        """Return a modified copy (validated again)."""
        return replace(self, **changes)


# Alias so ``SimulationConfig(...)`` and ``ScenarioConfig(...)`` are interchangeable.
SimulationConfig = ScenarioConfig
