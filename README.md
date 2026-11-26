# cold_chain_simulator

Generates **realistic, configurable, reproducible and labeled** cold-chain shipment
data for **PharmaGuard AI – Intelligent Cold-Chain Risk Detection and Pharmaceutical
Batch Protection**.

```
simulation → sensor data → ground-truth labels → DataFrame / CSV
```

This module only *produces* test data. It contains no detection, ML, dashboard,
database or API code. Dependencies: `numpy`, `pandas`. Python 3.10+.

## Install / import

Copy the `cold_chain_simulator/` folder next to your code, then:

```bash
pip install -r cold_chain_simulator/requirements.txt
```

```python
from cold_chain_simulator import ScenarioConfig, generate_scenario, generate_batch
```

## Quick start

```python
from cold_chain_simulator import ScenarioConfig, generate_scenario

config = ScenarioConfig(
    scenario_type="SUDDEN_SPIKE",
    duration_minutes=180,
    sampling_interval_minutes=1,
    random_seed=42,
)
df = generate_scenario(config)
df.to_csv("sudden_spike.csv", index=False)
```

Shortcuts: `generate_scenario(seed=42)` and keyword overrides such as
`generate_scenario(config, scenario_type="SENSOR_FAULT", fault_type="OFFLINE")`.
Run `python -m cold_chain_simulator.example` (from the folder containing the package)
for a full demo that writes CSVs to `simulator_output/`.

## Scenario types

| `scenario_type` | What happens | Ground-truth label |
|---|---|---|
| `NORMAL` | Temperature hovers around the target | `NORMAL` |
| `SUDDEN_SPIKE` | Short, sharp rise or drop (door open) that recovers smoothly | `TEMPERATURE_SPIKE` |
| `PROLONGED_EXCURSION` | Real temperature held outside the safe range for a long time | `PROLONGED_EXCURSION` |
| `SENSOR_FAULT` | Real temperature is fine; the sensor misreports (see fault types) | `SENSOR_FAULT` |
| `NOISY_SENSOR` | Real temperature stable; readings very noisy, zero-mean, no sustained shift | `NOISY_SENSOR` |
| `UNUSUAL_PATTERN` | Drift, repeated spikes, oscillation or rapid fluctuation, mostly inside the safe range | `UNUSUAL_PATTERN` |
| `SHIPMENT_DELAY` | Shipment takes longer than expected; optional slow warming while late | `SHIPMENT_DELAY` |

**Fault types** (`fault_type`): `FROZEN` (repeats last value), `MISSING` (random readings lost,
NaN), `UNREALISTIC` (impossible values like 85.0 / -127.0 / 125.0), `JUMP` (reading offset,
real temp fine), `OFFLINE` (no data for a stretch, NaN), `DRIFT` (calibration drift), or `RANDOM`.

**Pattern types** (`pattern_type`): `DRIFT`, `REPEATED_SPIKES`, `OSCILLATION`,
`RAPID_FLUCTUATION`, or `RANDOM`.

## Configuration (`ScenarioConfig`, alias `SimulationConfig`)

Every field has a default; change only what you need.

| Field | Default | Meaning |
|---|---|---|
| `shipment_id`, `batch_id` | `SHIP001`, `BATCH001` | Identifiers |
| `scenario_type` | `NORMAL` | One of the seven types |
| `duration_minutes` | 180 | Shipment length (the *actual* duration) |
| `sampling_interval_minutes` | 1 | Time between readings |
| `start_time` | 2025-01-01 08:00 | First timestamp |
| `target_temperature` | 5.0 | Setpoint (°C) |
| `lower_threshold`, `upper_threshold` | 2.0, 8.0 | Safe range |
| `starting_temperature` | `None` | `None` → target + small random offset |
| `noise_level` | 0.15 | Std-dev of normal sensor noise |
| `natural_variation` | 0.05 | Std-dev of real temperature change per step |
| `thermal_response` | 0.3 | How fast the real temperature follows its setpoint (0–1) |
| `cycle_amplitude` | 0.15 | Slow compressor-cycle swing |
| `random_seed` | 42 | Reproducibility seed |
| `event_start_minute` | `None` | `None` → random but reproducible |
| `event_duration_minutes` | `None` | `None` → scenario-specific random default (for `SHIPMENT_DELAY`: the delay length) |
| `severity` | 1.0 | Scales default event sizes |
| `magnitude` | `None` | Explicit event size in °C (overrides the severity default) |
| `direction` | `up` | `up`, `down` or `random` |
| `fault_type`, `pattern_type` | `FROZEN`, `DRIFT` | See above |
| `expected_duration_minutes` | `None` | `SHIPMENT_DELAY`: planned transit time |
| `delay_deterioration` | `True` | `SHIPMENT_DELAY`: temperature creeps up while late |
| `event_id` | `None` | `None` → `<shipment_id>-EVT01` |

## How the data is generated

```
temp(t)   = temp(t-1) + thermal_response * (setpoint(t) - temp(t-1)) + natural_variation
setpoint  = target + slow compressor cycle + scenario effect(t)
reading   = temp(t) + Gaussian sensor noise      (then any sensor fault is applied)
```

Each value builds on the previous one, and scenario effects use smooth ramps, so spikes
rise and recover over several samples instead of jumping between unrelated values.
Readings are rounded to 0.01 °C.

## Output schema

| Column | Description |
|---|---|
| `timestamp` | Reading time |
| `shipment_id`, `batch_id` | Identifiers |
| `temperature` | Sensor reading (°C). May be NaN (missing/offline) or an impossible value |
| `target_temperature`, `lower_threshold`, `upper_threshold` | Limits for this shipment |
| `sensor_status` | What a device would self-report: `OK`, `NO_DATA`, `OFFLINE`. Frozen, jump, drift and unrealistic faults are silent (`OK`) |
| `scenario_type` | Simulated scenario *(ground truth)* |
| `ground_truth_label` | Label of this record *(ground truth)* |
| `ground_truth_risk` | 0 normal, 1 inside an injected event/anomaly *(ground truth)* |
| `event_id` | Event identifier, empty for normal records *(ground truth)* |
| `true_temperature` | Real temperature without sensor noise/faults *(ground truth, extra)* |
| `elapsed_minutes` | Minutes since shipment start (extra) |
| `expected_duration_minutes`, `actual_duration_minutes` | Planned vs actual transit time; equal except in `SHIPMENT_DELAY` (extra) |

## Ground-truth labeling

Labels come **directly from the known simulation event**, never from a detector,
threshold check or statistics.

- Records inside the injected event get the scenario's label, `ground_truth_risk = 1` and the event's `event_id`. All other records are `NORMAL`, `0`, empty.
- Event window per scenario: spike / excursion / unusual pattern → the injected window **plus the short recovery tail** while the real temperature settles; sensor faults → the faulty window (for `MISSING` and `UNREALISTIC`, only the affected readings); noisy sensor → the noisy window; delay → every record after the planned arrival time.
- `ground_truth_risk = 1` means "inside a known anomaly/event", including sensor-quality events. Filter on `ground_truth_label` if you only want product-risk events (e.g. exclude `SENSOR_FAULT`, `NOISY_SENSOR`).
- `labels.strip_ground_truth(df)` removes all ground-truth columns (including `true_temperature`) so a detector can't peek.

## Reproducibility

The same config and seed always give identical output; different seeds give different data.

```python
a = generate_scenario(seed=42)
b = generate_scenario(seed=42)
assert a.equals(b)
```

`generate_batch(seed=...)` derives an independent seed for every shipment from the master seed.

## Generating CSV datasets

```python
from cold_chain_simulator import generate_batch

df = generate_batch(
    scenarios=["NORMAL", "SUDDEN_SPIKE", "PROLONGED_EXCURSION", "SENSOR_FAULT",
               "NOISY_SENSOR", "UNUSUAL_PATTERN", "SHIPMENT_DELAY"],
    samples_per_scenario=10,
    seed=42,
)
df.to_csv("dataset.csv", index=False)
```

Every shipment gets a unique `shipment_id` (`SHIP0001`…), `batch_id` and `event_id`.
By default (`randomize=True`) each shipment varies slightly in noise level, starting
temperature, severity and direction; `SENSOR_FAULT` and `UNUSUAL_PATTERN` shipments cycle through
all fault/pattern types. Use `base_config=ScenarioConfig(...)` for shared settings such as
thresholds or duration, or `randomize=False` to vary only seeds and IDs.

## Per-scenario examples

```python
cfg = ScenarioConfig(random_seed=1)
generate_scenario(cfg, scenario_type="SUDDEN_SPIKE", event_start_minute=60,
                  event_duration_minutes=6, magnitude=7.0, direction="up")
generate_scenario(cfg, scenario_type="PROLONGED_EXCURSION", direction="down")
generate_scenario(cfg, scenario_type="SENSOR_FAULT", fault_type="UNREALISTIC")
generate_scenario(cfg, scenario_type="NOISY_SENSOR", severity=1.5)
generate_scenario(cfg, scenario_type="UNUSUAL_PATTERN", pattern_type="OSCILLATION")
generate_scenario(cfg, scenario_type="SHIPMENT_DELAY", expected_duration_minutes=120)
```

## Module layout

| File | Role |
|---|---|
| `config.py` | `ScenarioConfig` dataclass and constants |
| `generator.py` | Base time-series model and `generate_scenario` |
| `scenarios.py` | The seven scenario definitions |
| `noise.py` | Reusable noise, spike, missing, frozen and drift functions |
| `labels.py` | Ground-truth labels from the known event |
| `batch_generator.py` | `generate_batch` for many shipments |
| `example.py` | Runnable demo |

### How this module connects to the detection system

The DataFrame returned by `generate_scenario` / `generate_batch` is the **input to the
rule-based and ML detection modules**: pass `strip_ground_truth(df)` (or just the
sensor columns) to the detector. `ground_truth_label`, `ground_truth_risk` and `event_id`
are **retained for later evaluation**: after the detector produces its predictions, join
them back on the same rows and compare against these labels (the evaluation itself lives
in the detection project, not here).
