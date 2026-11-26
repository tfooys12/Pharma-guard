"""Cold-chain scenario simulator: labeled synthetic shipment data for PharmaGuard AI."""
from .batch_generator import generate_batch
from .config import (FAULT_TYPES, PATTERN_TYPES, SCENARIO_TYPES,
                     ScenarioConfig, SimulationConfig)
from .generator import generate_scenario
from .labels import GROUND_TRUTH_COLUMNS, LABELS, strip_ground_truth

__version__ = "1.0.0"

__all__ = [
    "ScenarioConfig", "SimulationConfig", "generate_scenario", "generate_batch",
    "strip_ground_truth", "SCENARIO_TYPES", "FAULT_TYPES", "PATTERN_TYPES",
    "LABELS", "GROUND_TRUTH_COLUMNS",
]
