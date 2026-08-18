# Monitoring Guide — prediction logging & data drift

A trained model is only valid while production data resembles the data it learned
from. When inputs shift ("data drift"), accuracy decays silently — no error is
thrown, the model just quietly gets worse. This is the layer that catches that.

> **The one idea:** drift detection answers "does *today's* input look like the data
> I trained on?" If not, the model's metrics no longer mean what you think.

---

## 1. Prediction logging

Every `/predict` call appends a JSON line to `monitoring/predictions.jsonl`
(timestamp, features, probability, decision). That log is the raw material for drift
checks and an audit trail of what the model actually did in production. Logging is
best-effort — a logging failure never breaks a prediction.

---

## 2. Two drift metrics (and why both)

We score each feature two complementary ways and flag it if **either** fires.

### PSI — Population Stability Index
A single number summarizing how much a feature's distribution moved. We bin the
reference (training) values into deciles and compare the proportion of current data
falling in each bin:

```
PSI = Σ (cur% − ref%) · ln(cur% / ref%)
```

Industry rule of thumb:

| PSI | Interpretation |
|---|---|
| < 0.10 | no significant shift |
| 0.10 – 0.25 | moderate shift — keep an eye on it |
| > 0.25 | **significant drift** — investigate |

### KS test — Kolmogorov–Smirnov
A statistical hypothesis test. Null hypothesis: the two samples come from the same
distribution. A **low p-value (< 0.05)** means that's unlikely → the distributions
differ. Unlike PSI it needs no binning, but it says nothing about *magnitude*.

**Why both:** PSI gives an interpretable magnitude; KS gives statistical rigor. PSI
can be noisy on small samples; KS can flag tiny, practically-irrelevant differences
on huge samples. Using both, with an OR rule, is more robust than either alone.

---

## 3. ⚠️ A real lesson: drift detection is unreliable on small samples

The first version of the live demo flagged **28/30 features as drifted on perfectly
normal data** — a false alarm. Two causes, both classic:

1. **Non-representative sample.** The inputs were the *first* N rows of an unshuffled
   dataset (a biased slice), not a random sample. Fix: sample randomly.
2. **Too few samples for PSI.** With only 30 inputs across 10 deciles, most bins are
   nearly empty; the zero-flooring then inflates PSI to absurd values (~12). Fix:
   require an adequate sample (`MIN_DRIFT_SAMPLES = 100`) before reporting.

After fixing both, normal traffic correctly reads **0/30 drifted** and inputs scaled
by ×1.4 read **30/30 drifted**. The takeaway for interviews: *drift monitoring is only
as trustworthy as your sample size and sampling method* — small-window drift alarms
are mostly noise.

---

## 4. How to use it

**Live endpoint** (on the running API):
```
GET /monitoring/drift            # check all logged predictions
GET /monitoring/drift?window=200 # check only the last 200
```
Returns per-feature PSI + KS p-value, sorted most-drifted first, plus a summary.
Below `MIN_DRIFT_SAMPLES` it returns `status: insufficient_data` instead of guessing.

**Scheduled / CI check** (`src/check_drift.py`):
```
python src/check_drift.py --window 500
```
Prints a report and **exits non-zero if drift is detected**, so a cron job or alerting
system can trigger on the exit code (e.g. page someone, or kick off retraining).

---

## 5. What a real system does next

This project detects drift; a production system would *act* on it:
- **Alert** when `share_drifted` crosses a threshold.
- **Trigger retraining** automatically (close the loop back to `train.py`).
- **Monitor prediction drift too** (the output distribution, not just inputs) and,
  once labels arrive, **true performance** (the real recall on malignant over time).
- **Segment by cohort** so drift in one slice isn't hidden by a stable average.

See [[evaluation]] for why recall on malignant is the metric that ultimately matters.
