import numpy as np
import subprocess
import sys

from random_variable_distribution import analyze_distribution, expected_probabilities, generate_uniform, transform_sample


def test_middle_square_reproducible():
    seq = generate_uniform(5, method="middle-square", seed=0.84616823, k=8)
    # deterministic first values
    assert np.allclose(seq[:3], [0.84616823, 0.00673461, 0.45354971], atol=1e-8)
    assert (seq >= 0).all() and (seq < 1).all()


def test_expected_probabilities_sum_to_one():
    probs = expected_probabilities(10)
    assert len(probs) == 10
    assert np.isclose(np.sum(probs), 1.0, atol=1e-6)


def test_analyze_distribution_returns_reasonable_chi_square():
    summary = analyze_distribution(size=2000, num_bins=12, method="numpy", seed=42)
    assert summary.samples.shape == (2000,)
    assert summary.bin_counts.size == 12
    # Chi-square should be finite and not extreme for deterministic seed
    assert summary.chi_square > 0
    assert summary.chi_square < 30
    assert summary.p_value is None or 0 <= summary.p_value <= 1


def test_transform_sample_matches_manual():
    uniforms = np.array([0.0, 0.5, 1.0])
    transformed = transform_sample(uniforms)
    expected = np.arcsin(np.sqrt(uniforms / 2))
    assert np.allclose(transformed, expected)


def test_cli_runs(tmp_path):
    cmd = [sys.executable, "-m", "random_variable_distribution", "-n", "100", "-b", "8", "-g", "numpy", "--seed", "123"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Generator: numpy" in result.stdout
    assert "Chi-square" in result.stdout
