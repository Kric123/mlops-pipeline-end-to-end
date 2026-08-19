"""MIN_DRIFT_SAMPLES is justified, not guessed.

Locks the finding from src/sample_size_study.py: at 30 rows the drift check
false-alarms on clean data most of the time, and at MIN_DRIFT_SAMPLES it does not.
If someone lowers the floor, this fails.
"""

from src.sample_size_study import false_alarm_rate
from src.serve import MIN_DRIFT_SAMPLES


def test_thirty_rows_false_alarms_on_clean_data():
    """The 28/30 incident: sample size alone trips the alarm."""
    r = false_alarm_rate(30, trials=2000)
    assert r["false_alarm_rate"] > 0.5, r
    assert r["p_empty_bin"] > 0.2, r


def test_min_drift_samples_keeps_false_alarms_under_one_percent():
    r = false_alarm_rate(MIN_DRIFT_SAMPLES, trials=2000)
    assert r["false_alarm_rate"] < 0.01, r


def test_false_alarm_rate_falls_monotonically_with_n():
    rates = [false_alarm_rate(n, trials=1000)["false_alarm_rate"]
             for n in (30, 50, 100, 200)]
    assert rates == sorted(rates, reverse=True), rates
