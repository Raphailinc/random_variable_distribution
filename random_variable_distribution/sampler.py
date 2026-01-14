from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover - SciPy optional
    chi2 = None


MIDDLE_SQUARE_SEED = 0.84616823
MIDDLE_SQUARE_K = 8


@dataclass
class SampleSummary:
    samples: np.ndarray
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    expected_probabilities: np.ndarray
    expected_counts: np.ndarray
    chi_square: float
    p_value: Optional[float]
    generator: str

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples))


def generate_uniform(
    size: int,
    *,
    method: str = "numpy",
    seed: Optional[float | int] = None,
    k: int = MIDDLE_SQUARE_K,
) -> np.ndarray:
    """
    Generate `size` uniform values on [0, 1).

    method:
      - "numpy": NumPy default_rng (seeded if provided).
      - "middle-square": middle-square PRNG with fractional seed (0 < seed < 1).
    """
    if size <= 0:
        raise ValueError("size must be positive")

    if method == "numpy":
        seed_to_use = seed
        if isinstance(seed_to_use, float) and float(seed_to_use).is_integer():
            seed_to_use = int(seed_to_use)
        rng = np.random.default_rng(seed_to_use)
        return rng.random(size)

    if method == "middle-square":
        return _middle_square(size, seed or MIDDLE_SQUARE_SEED, k=k)

    raise ValueError(f"Unknown method: {method}")


def _middle_square(size: int, seed: float, *, k: int) -> np.ndarray:
    if not (0 < seed < 1):
        raise ValueError("seed must be between 0 and 1 (exclusive) for middle-square generator")
    if k <= 0 or k % 2 != 0:
        raise ValueError("k must be a positive even integer")

    seq = np.empty(size)
    seq[0] = seed
    scale_half = 10 ** (k // 2)
    scale = 10 ** k
    for idx in range(1, size):
        val = seq[idx - 1] ** 2
        fractional = math.modf(val * scale_half)[0]
        next_val = int(fractional * scale) / scale
        seq[idx] = next_val
    return seq


def transform_sample(uniforms: Iterable[float]) -> np.ndarray:
    """Apply inverse CDF: X = asin(sqrt(U/2))."""
    arr = np.asarray(list(uniforms), dtype=float)
    return np.arcsin(np.sqrt(arr / 2))


def expected_probabilities(num_bins: int) -> np.ndarray:
    """Theoretical probability for each bin over [0, pi/4] split into num_bins parts."""
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    delta = (math.pi / 4) / num_bins
    probs = []
    for i in range(1, num_bins + 1):
        x_i = i * delta
        x_prev = (i - 1) * delta
        p_i = 2 * (math.sin(x_i) ** 2) - 2 * (math.sin(x_prev) ** 2)
        probs.append(p_i)
    return np.array(probs, dtype=float)


def analyze_distribution(
    *,
    size: int = 5000,
    num_bins: int = 14,
    method: str = "numpy",
    seed: Optional[float | int] = None,
    k: int = MIDDLE_SQUARE_K,
) -> SampleSummary:
    uniforms = generate_uniform(size, method=method, seed=seed, k=k)
    samples = transform_sample(uniforms)
    bin_edges = np.linspace(0, math.pi / 4, num_bins + 1)
    bin_counts, _ = np.histogram(samples, bins=bin_edges)
    probs = expected_probabilities(num_bins)
    expected_counts = probs * size
    chi_square = float(np.sum((bin_counts - expected_counts) ** 2 / expected_counts))

    p_value = None
    if chi2:
        dof = num_bins - 1
        p_value = float(chi2.sf(chi_square, dof))

    return SampleSummary(
        samples=samples,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        expected_probabilities=probs,
        expected_counts=expected_counts,
        chi_square=chi_square,
        p_value=p_value,
        generator=method,
    )


def _format_summary(summary: SampleSummary, top_n: int = 8) -> str:
    lines = [
        f"Generator: {summary.generator}",
        f"Samples: {len(summary.samples)}, bins: {len(summary.bin_counts)}",
        f"Mean: {summary.mean:.6f}, Std: {summary.std:.6f}",
        f"Chi-square: {summary.chi_square:.4f}"
        + (f", p-value: {summary.p_value:.4f}" if summary.p_value is not None else ""),
        f"First {top_n} samples: " + ", ".join(f"{x:.6f}" for x in summary.samples[:top_n]),
        "Bin counts (observed): " + ", ".join(str(int(x)) for x in summary.bin_counts),
        "Bin counts (expected): " + ", ".join(f"{x:.1f}" for x in summary.expected_counts),
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Sample and analyse a random variable distribution.")
    parser.add_argument("-n", "--size", type=int, default=5000, help="Number of samples to generate.")
    parser.add_argument("-b", "--bins", type=int, default=14, help="Number of histogram bins.")
    parser.add_argument(
        "-g",
        "--generator",
        choices=["numpy", "middle-square"],
        default="numpy",
        help="Uniform generator to use.",
    )
    parser.add_argument("--seed", type=float, default=None, help="Seed for generator.")
    parser.add_argument("--k", type=int, default=MIDDLE_SQUARE_K, help="k parameter for middle-square (even).")
    args = parser.parse_args(argv)

    summary = analyze_distribution(
        size=args.size,
        num_bins=args.bins,
        method=args.generator,
        seed=args.seed,
        k=args.k,
    )
    print(_format_summary(summary))


if __name__ == "__main__":  # pragma: no cover
    main()
