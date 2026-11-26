"""Reusable sensor-noise helpers.

All functions are pure: they take a NumPy ``Generator`` (or plain arrays) and
return new arrays, never touching global random state. That keeps every
scenario reproducible from its seed.
"""
from __future__ import annotations

import numpy as np


def gaussian_noise(rng: np.random.Generator, n: int, std: float) -> np.ndarray:
    """Normal measurement noise, zero mean. ``std`` may also be an array of length n."""
    return rng.normal(0.0, std, n)


def high_noise(
    rng: np.random.Generator, n: int, std: float, mask: np.ndarray | None = None
) -> np.ndarray:
    """Extra zero-mean noise with a large ``std``, only inside ``mask`` (or everywhere).

    Meant to be *added on top of* the normal Gaussian noise.
    """
    out = rng.normal(0.0, std, n)
    return out if mask is None else np.where(mask, out, 0.0)


def random_spikes(
    rng: np.random.Generator,
    n: int,
    probability: float,
    magnitude: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Isolated one-sample outliers of random sign, size ~0.5-1.5 x ``magnitude``."""
    hits = rng.random(n) < probability
    if mask is not None:
        hits &= mask
    signs = rng.choice([-1.0, 1.0], n)
    sizes = magnitude * (0.5 + rng.random(n))
    return np.where(hits, signs * sizes, 0.0)


def missing_mask(
    rng: np.random.Generator, n: int, probability: float, window: np.ndarray | None = None
) -> np.ndarray:
    """Boolean mask of readings that are lost (optionally only inside ``window``)."""
    hits = rng.random(n) < probability
    return hits if window is None else hits & window


def frozen_values(values: np.ndarray, start: int, end: int) -> np.ndarray:
    """Copy of ``values`` where [start, end) repeats the last good reading."""
    out = values.copy()
    out[start:end] = values[max(start - 1, 0)]
    return out


def sensor_drift(n: int, start: int, end: int, rate_per_sample: float) -> np.ndarray:
    """Additive calibration drift: grows linearly inside [start, end), zero elsewhere."""
    out = np.zeros(n)
    out[start:end] = rate_per_sample * np.arange(1, end - start + 1)
    return out


def sensor_jump(n: int, start: int, end: int, offset: float) -> np.ndarray:
    """Additive constant offset inside [start, end): a sudden jump in the reading only."""
    out = np.zeros(n)
    out[start:end] = offset
    return out
