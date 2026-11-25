"""Unbiased pass@k estimator from the HumanEval paper (Chen et al., 2021)."""

from __future__ import annotations


def pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Probability that >=1 of k random samples (drawn without replacement
    from num_samples) is correct, given num_correct correct samples overall.

    pass@k = 1 - C(n - c, k) / C(n, k), computed in a numerically stable form.
    """
    if k <= 0:
        raise ValueError("k must be >= 1.")
    if num_samples < 0 or num_correct < 0 or num_correct > num_samples:
        raise ValueError(
            "Invalid (num_samples={}, num_correct={}).".format(num_samples, num_correct)
        )
    if k > num_samples:
        raise ValueError(
            "k={} exceeds num_samples={}; cannot compute pass@k.".format(k, num_samples)
        )
    if num_samples - num_correct < k:
        return 1.0
    result = 1.0
    for i in range(num_samples - num_correct + 1, num_samples + 1):
        result *= 1.0 - k / i
    return 1.0 - result
