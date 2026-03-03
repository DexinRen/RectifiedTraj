#!/usr/bin/env python3
"""BlogWatcher mimic interval distribution fitting and sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Dict

import numpy as np


SQRT2 = sqrt(2.0)


@dataclass(frozen=True)
class MixtureLognormalParams:
    """Parameters for a two-component lognormal mixture."""

    weight_body: float
    body_mu: float
    body_sigma: float
    tail_mu: float
    tail_sigma: float


@dataclass(frozen=True)
class FitResult:
    """Best-fit parameters and achieved summary statistics."""

    params: MixtureLognormalParams
    target_mean: float
    target_median: float
    target_std: float
    achieved_mean: float
    achieved_median: float
    achieved_std: float
    score: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "weight_body": self.params.weight_body,
            "body_mu": self.params.body_mu,
            "body_sigma": self.params.body_sigma,
            "tail_mu": self.params.tail_mu,
            "tail_sigma": self.params.tail_sigma,
            "target_mean": self.target_mean,
            "target_median": self.target_median,
            "target_std": self.target_std,
            "achieved_mean": self.achieved_mean,
            "achieved_median": self.achieved_median,
            "achieved_std": self.achieved_std,
            "score": self.score,
        }


def _lognormal_raw_moments(mu: float, sigma: float) -> tuple[float, float]:
    m1 = exp(mu + 0.5 * sigma * sigma)
    m2 = exp(2.0 * mu + 2.0 * sigma * sigma)
    return m1, m2


def _mixture_mean_std(params: MixtureLognormalParams) -> tuple[float, float]:
    w = params.weight_body
    m1_b, m2_b = _lognormal_raw_moments(params.body_mu, params.body_sigma)
    m1_t, m2_t = _lognormal_raw_moments(params.tail_mu, params.tail_sigma)

    mean = w * m1_b + (1.0 - w) * m1_t
    second = w * m2_b + (1.0 - w) * m2_t
    var = max(second - mean * mean, 0.0)
    return mean, sqrt(var)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / SQRT2))


def _mixture_cdf(x: float, params: MixtureLognormalParams) -> float:
    if x <= 0.0:
        return 0.0
    lx = log(x)
    z1 = (lx - params.body_mu) / params.body_sigma
    z2 = (lx - params.tail_mu) / params.tail_sigma
    w = params.weight_body
    return w * _normal_cdf(z1) + (1.0 - w) * _normal_cdf(z2)


def _mixture_median(params: MixtureLognormalParams) -> float:
    # Find a robust upper bound with expanding search.
    lo = 1e-6
    hi = max(exp(params.tail_mu + 8.0 * params.tail_sigma), exp(params.body_mu + 6.0 * params.body_sigma), 1.0)
    for _ in range(40):
        if _mixture_cdf(hi, params) >= 0.5:
            break
        hi *= 2.0

    # Bisection for median.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _mixture_cdf(mid, params) < 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _score(
    target_mean: float,
    target_median: float,
    target_std: float,
    achieved_mean: float,
    achieved_median: float,
    achieved_std: float,
) -> float:
    # Median is weighted slightly higher so we preserve central cadence.
    e_mean = (achieved_mean - target_mean) / max(target_mean, 1e-9)
    e_med = (achieved_median - target_median) / max(target_median, 1e-9)
    e_std = (achieved_std - target_std) / max(target_std, 1e-9)
    return e_mean * e_mean + 1.5 * e_med * e_med + e_std * e_std


def fit_two_lognormal_mixture(
    target_mean: float,
    target_median: float,
    target_std: float,
    *,
    seed: int = 42,
    n_iter: int = 16000,
) -> FitResult:
    """Fit a two-lognormal mixture to target mean/median/std via random search.

    The search space is tuned for highly-skewed interval distributions where
    the median is much smaller than the mean.
    """

    if target_mean <= 0 or target_median <= 0 or target_std < 0:
        raise ValueError("Target statistics must satisfy: mean>0, median>0, std>=0")

    rng = np.random.default_rng(seed)

    best_params = None
    best_score = float("inf")
    best_stats = (0.0, 0.0, 0.0)

    log_med = log(target_median)

    global_iters = max(1000, int(n_iter * 0.7))
    local_iters = max(500, n_iter - global_iters)

    for _ in range(global_iters):
        # Body captures common shorter intervals.
        body_sigma = float(rng.uniform(0.20, 1.20))
        body_mu = float(log_med + rng.normal(0.0, 0.35))

        # Tail captures rare large gaps.
        tail_sigma = float(rng.uniform(0.35, 2.20))
        tail_mu = float(log_med + rng.uniform(1.0, 7.0))

        # Body usually dominates count so the median stays realistic.
        weight_body = float(rng.uniform(0.55, 0.98))

        params = MixtureLognormalParams(
            weight_body=weight_body,
            body_mu=body_mu,
            body_sigma=body_sigma,
            tail_mu=tail_mu,
            tail_sigma=tail_sigma,
        )

        achieved_mean, achieved_std = _mixture_mean_std(params)
        achieved_median = _mixture_median(params)
        s = _score(
            target_mean,
            target_median,
            target_std,
            achieved_mean,
            achieved_median,
            achieved_std,
        )

        if s < best_score:
            best_params = params
            best_score = s
            best_stats = (achieved_mean, achieved_median, achieved_std)

    # Local refinement around the best global candidate.
    assert best_params is not None
    for _ in range(local_iters):
        weight_body = float(np.clip(best_params.weight_body + rng.normal(0.0, 0.03), 0.55, 0.995))
        body_mu = float(best_params.body_mu + rng.normal(0.0, 0.25))
        tail_mu = float(best_params.tail_mu + rng.normal(0.0, 0.35))
        body_sigma = float(np.clip(best_params.body_sigma * exp(rng.normal(0.0, 0.18)), 0.12, 2.5))
        tail_sigma = float(np.clip(best_params.tail_sigma * exp(rng.normal(0.0, 0.22)), 0.12, 3.2))

        params = MixtureLognormalParams(
            weight_body=weight_body,
            body_mu=body_mu,
            body_sigma=body_sigma,
            tail_mu=tail_mu,
            tail_sigma=tail_sigma,
        )

        achieved_mean, achieved_std = _mixture_mean_std(params)
        achieved_median = _mixture_median(params)
        s = _score(
            target_mean,
            target_median,
            target_std,
            achieved_mean,
            achieved_median,
            achieved_std,
        )
        if s < best_score:
            best_params = params
            best_score = s
            best_stats = (achieved_mean, achieved_median, achieved_std)

    return FitResult(
        params=best_params,
        target_mean=target_mean,
        target_median=target_median,
        target_std=target_std,
        achieved_mean=best_stats[0],
        achieved_median=best_stats[1],
        achieved_std=best_stats[2],
        score=best_score,
    )


class SampleTimeGenerator:
    """Reproducible sampler for fitted inter-sample-time distribution."""

    def __init__(self, params: MixtureLognormalParams, seed: int = 0) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    def sample(self, size: int, *, min_seconds: float = 1.0, round_to_int: bool = False) -> np.ndarray:
        if size <= 0:
            return np.empty((0,), dtype=np.float64)

        w = self.params.weight_body
        use_body = self.rng.random(size) < w

        out = np.empty(size, dtype=np.float64)
        n_body = int(use_body.sum())
        n_tail = int(size - n_body)

        if n_body > 0:
            out[use_body] = self.rng.lognormal(mean=self.params.body_mu, sigma=self.params.body_sigma, size=n_body)
        if n_tail > 0:
            out[~use_body] = self.rng.lognormal(mean=self.params.tail_mu, sigma=self.params.tail_sigma, size=n_tail)

        out = np.maximum(out, float(min_seconds))
        if round_to_int:
            out = np.rint(out)
        return out


def summarize_samples(samples: np.ndarray) -> Dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return {"n": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "n": float(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def build_blogwatch_sample_time_generator(
    *,
    target_mean: float = 867.618530,
    target_median: float = 128.0,
    target_std: float = 4773.800293,
    fit_seed: int = 42,
    sample_seed: int = 7,
    n_iter: int = 20000,
) -> tuple[FitResult, SampleTimeGenerator]:
    """Build a reproducible generator for BlogWatcher-mimic sample intervals."""

    fit = fit_two_lognormal_mixture(
        target_mean=target_mean,
        target_median=target_median,
        target_std=target_std,
        seed=fit_seed,
        n_iter=n_iter,
    )
    generator = SampleTimeGenerator(fit.params, seed=sample_seed)
    return fit, generator


if __name__ == "__main__":
    fit, gen = build_blogwatch_sample_time_generator()
    draws = gen.sample(200000, min_seconds=1.0, round_to_int=True)

    print("Best fit:")
    for k, v in fit.as_dict().items():
        print(f"  {k}: {v}")

    print("\nSample check:")
    for k, v in summarize_samples(draws).items():
        print(f"  {k}: {v}")
