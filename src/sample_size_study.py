"""Why MIN_DRIFT_SAMPLES is 100.

The live demo once flagged 28/30 features as drifted on data that had not moved
(docs/monitoring.md, section 3). Two causes: a non-random sample, and too few rows
for decile binning. The sampling bug is fixed. This measures the second cause, so
the sample-size floor is a number we chose rather than one we guessed.

Method: draw `n` rows from the *reference distribution itself* — no drift, by
construction — bin on reference deciles, and score with the same
population_stability_index() the endpoint uses, 1e-6 floor included. Any PSI above
PSI_THRESHOLD here is a false alarm produced by sample size alone.

Run: python -m src.sample_size_study
"""

from __future__ import annotations

import numpy as np

from src.monitoring import PSI_THRESHOLD, population_stability_index

BINS = 10
TRIALS = 5_000
SIZES = (30, 50, 100, 200, 500)
SEED = 7


def false_alarm_rate(n: int, trials: int = TRIALS, bins: int = BINS,
                     seed: int = SEED) -> dict:
    """PSI of a no-drift sample of size `n` against its own reference."""
    rng = np.random.default_rng(seed)
    reference = rng.standard_normal(100_000)
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    psi = np.empty(trials)
    empty = np.zeros(trials, dtype=bool)
    for i in range(trials):
        # iid draw from the reference itself: no drift by construction
        current = reference[rng.integers(0, reference.size, size=n)]
        psi[i] = population_stability_index(reference, current, bins=bins)
        empty[i] = (np.histogram(current, bins=edges)[0] == 0).any()
    return {
        "n": n,
        "mean_psi": float(psi.mean()),
        "median_psi": float(np.median(psi)),
        "false_alarm_rate": float((psi > PSI_THRESHOLD).mean()),
        "p_empty_bin": float(empty.mean()),
    }


def main() -> None:
    print(f"No-drift PSI by sample size  (bins={BINS}, threshold={PSI_THRESHOLD}, "
          f"{TRIALS:,} trials each)\n")
    print(f"{'n':>5} {'mean PSI':>10} {'median':>9} {'false alarms':>14} {'empty bin':>11}")
    for n in SIZES:
        r = false_alarm_rate(n)
        print(f"{r['n']:>5} {r['mean_psi']:>10.3f} {r['median_psi']:>9.3f} "
              f"{r['false_alarm_rate']:>13.1%} {r['p_empty_bin']:>10.1%}")
    print("\nEmpty bins are the mechanism: the 1e-6 floor turns each one into a")
    print("contribution of (1e-6 - 0.1) * ln(1e-6 / 0.1) = 1.151 on its own.")
    print("Nine empty bins - all rows in one decile - gives PSI = 12.4, ~50x threshold.")
    print("That is the 28/30 incident. MIN_DRIFT_SAMPLES = 100 is set here.")


if __name__ == "__main__":
    main()
