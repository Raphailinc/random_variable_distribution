"""
Toolkit for sampling and analysing a custom random variable distribution.

Exposes helper functions:
- generate_uniform(): produce base uniforms with either numpy RNG or middle-square PRNG.
- transform_sample(): apply the inverse CDF to uniforms.
- analyze_distribution(): run sampling, binning, chi-square test and return a summary object.
"""

from .sampler import (
    SampleSummary,
    analyze_distribution,
    expected_probabilities,
    generate_uniform,
    transform_sample,
)

__all__ = [
    "SampleSummary",
    "analyze_distribution",
    "expected_probabilities",
    "generate_uniform",
    "transform_sample",
]
