"""Runnable demo of the cold-chain simulator API.

Run from the folder that CONTAINS ``cold_chain_simulator/``:

    python -m cold_chain_simulator.example
    # or: python cold_chain_simulator/example.py

CSV files are written to ``./simulator_output/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Lets ``python cold_chain_simulator/example.py`` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cold_chain_simulator import ScenarioConfig, generate_batch, generate_scenario  # noqa: E402

OUT_DIR = Path("simulator_output")


def summarize(name: str, df) -> None:
    flagged = int(df["ground_truth_risk"].sum())
    print(f"{name:<22} rows={len(df):4d}  risk records={flagged:3d}  "
          f"temp range=({df['temperature'].min():.1f}, {df['temperature'].max():.1f})")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Create a configuration (every field has a default and is easy to change)
    config = ScenarioConfig(
        shipment_id="SHIP001",
        batch_id="BATCH001",
        duration_minutes=180,
        sampling_interval_minutes=1,
        target_temperature=5.0,
        lower_threshold=2.0,
        upper_threshold=8.0,
        random_seed=42,
    )

    # 2. Normal shipment
    normal = generate_scenario(config)
    summarize("NORMAL", normal)

    # 3. Sudden spike - override any config field with keyword arguments
    spike = generate_scenario(config, scenario_type="SUDDEN_SPIKE",
                              event_start_minute=60, event_duration_minutes=6, direction="up")
    summarize("SUDDEN_SPIKE", spike)

    # 4. Prolonged excursion (low side this time)
    excursion = generate_scenario(config, scenario_type="PROLONGED_EXCURSION",
                                  direction="down", event_start_minute=50, event_duration_minutes=60)
    summarize("PROLONGED_EXCURSION", excursion)

    # 5. Sensor fault - choose the fault type
    fault = generate_scenario(config, scenario_type="SENSOR_FAULT", fault_type="FROZEN")
    summarize("SENSOR_FAULT (frozen)", fault)

    # 6. Noisy sensor - real temperature stays stable, readings are noisy
    noisy = generate_scenario(config, scenario_type="NOISY_SENSOR")
    summarize("NOISY_SENSOR", noisy)

    # 7. Unusual pattern - e.g. OSCILLATION, DRIFT, REPEATED_SPIKES, RAPID_FLUCTUATION
    unusual = generate_scenario(config, scenario_type="UNUSUAL_PATTERN", pattern_type="OSCILLATION")
    summarize("UNUSUAL_PATTERN", unusual)

    # 8. Shipment delay - expected vs actual timing are stored in the data
    delay = generate_scenario(config, scenario_type="SHIPMENT_DELAY", expected_duration_minutes=120)
    summarize("SHIPMENT_DELAY", delay)
    print("   expected/actual duration (min):",
          delay["expected_duration_minutes"].iloc[0], "/", delay["actual_duration_minutes"].iloc[0])

    # 9. Generate a batch of many scenarios
    batch = generate_batch(
        scenarios=["NORMAL", "SUDDEN_SPIKE", "PROLONGED_EXCURSION", "SENSOR_FAULT",
                   "NOISY_SENSOR", "UNUSUAL_PATTERN", "SHIPMENT_DELAY"],
        samples_per_scenario=10,
        seed=42,
    )
    print(f"\nBatch: {len(batch)} rows, {batch['shipment_id'].nunique()} shipments")
    print(batch.groupby("scenario_type")["ground_truth_risk"].mean().round(2).to_string())

    # 10. Save to CSV
    normal.to_csv(OUT_DIR / "normal.csv", index=False)
    spike.to_csv(OUT_DIR / "sudden_spike.csv", index=False)
    batch.to_csv(OUT_DIR / "batch_dataset.csv", index=False)
    print(f"\nSaved CSV files to {OUT_DIR.resolve()}")

    print("\nFirst rows of the spike scenario:")
    print(spike.head().to_string(index=False))


if __name__ == "__main__":
    main()
