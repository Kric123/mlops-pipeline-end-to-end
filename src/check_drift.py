"""
CLI drift check — compare logged predictions against the training reference.

Designed to run on a schedule (cron / CI). Prints a report and exits non-zero
if drift is detected, so an alerting system can act on the exit code.

Run:
    python src/check_drift.py            # check all logged predictions
    python src/check_drift.py --window 200
"""

from __future__ import annotations

import argparse
import sys

try:
    from src import monitoring
except ImportError:  # script-mode fallback
    import monitoring


def main() -> int:
    parser = argparse.ArgumentParser(description="Data drift check")
    parser.add_argument("--window", type=int, default=None,
                        help="Only check the last N logged predictions.")
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()

    current = monitoring.load_recent_predictions(args.window)
    if len(current) < args.min_samples:
        print(f"Insufficient data: {len(current)} predictions "
              f"(need >= {args.min_samples}).")
        return 0  # not an error, just nothing to check yet

    reference = monitoring.load_reference()
    report = monitoring.compute_drift_report(reference, current)
    s = report["summary"]

    print(f"Checked {len(current)} predictions against {s['n_features']} features.")
    print(f"Drifted features: {s['n_drifted']}/{s['n_features']} "
          f"({s['share_drifted']:.0%})")
    if s["overall_drift"]:
        print("\nTop drifted features (PSI | KS p-value):")
        for f in report["features"]:
            if f["drifted"]:
                print(f"  {f['feature']:<28} PSI={f['psi']:>7.3f}  ks_p={f['ks_p_value']:.4f}")
        print("\nDRIFT DETECTED — investigate / consider retraining.")
        return 1

    print("No significant drift.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
