# Evaluation Guide — what each metric means and when it matters

This document explains the metrics logged by `src/train.py`, why each one exists,
how to read it, and — most importantly — **which metric you should actually optimize
for this problem**. The numbers below are from the M1 run with **malignant as the
positive class** (2026-06-19). Precision/recall/F1 are therefore measured for *detecting
malignant tumors*.

> **The one idea to internalize:** there is no single "best" metric. The right metric
> is decided by *what a mistake costs* in your problem. Picking the metric is a
> modeling decision, not a formula.

---

## 0. The foundation: the confusion matrix

Every classification metric is just a different ratio built from four counts. For a
binary problem you pick one class as **positive** (the thing you're trying to detect),
then every prediction falls into one of four boxes:

|                   | Predicted Positive | Predicted Negative |
|-------------------|--------------------|--------------------|
| **Actually Positive** | True Positive (TP)  | False Negative (FN) ← *missed it* |
| **Actually Negative** | False Positive (FP) ← *false alarm* | True Negative (TN)  |

- **FN = a miss** (you said "no" but it was "yes")
- **FP = a false alarm** (you said "yes" but it was "no")

Almost every metric is a trade-off between how much you fear FNs vs. FPs.

---

## 1. Accuracy — `0.9649`

**Formula:** `(TP + TN) / everything`
**Plain meaning:** of all predictions, what fraction were correct.

**When it's useful:** balanced classes and roughly equal cost of both error types.

**The trap:** accuracy lies on imbalanced data. If 99% of emails are not-spam, a model
that predicts "not-spam" *every time* scores 99% accuracy while being useless. **Always
sanity-check accuracy against the class balance.** Our dataset is ~63% benign / 37%
malignant — mildly imbalanced — so accuracy is *informative but not sufficient* here.

---

## 2. Precision (malignant) — `1.0000`

**Formula:** `TP / (TP + FP)`
**Plain meaning:** of everything the model *flagged as positive*, what fraction really was.
**Answers:** "When it says yes, how often is it right?"

**Optimize precision when a false alarm is expensive.** Examples: flagging a transaction
as fraud (and freezing a real customer's card), recommending content, spam filters
(don't send a real email to junk). High precision = few false alarms.

---

## 3. Recall / Sensitivity (malignant) — `0.9048`  ⚠️ the one to watch

**Formula:** `TP / (TP + FN)`
**Plain meaning:** of all the *actual positives*, what fraction did the model catch.
**Answers:** "Of the real cases out there, how many did we find?"

**Optimize recall when a miss is expensive.** Examples: disease screening (don't miss a
sick patient), security threat detection, predictive maintenance (don't miss the failure).
High recall = few misses (few FNs).

> **Our result:** recall on malignant is only **0.9048** — the model misses **4 of 42**
> cancers (FN = 4) at the default 0.5 threshold. This was *hidden* when we measured recall
> on the benign class (0.972). It's the single most important number for a screening tool,
> and §8 / the threshold sweep show how to push it up.

---

## 4. The precision↔recall trade-off (the heart of it)

Precision and recall **pull against each other**. A classifier outputs a *probability*;
you convert it to yes/no with a **threshold** (default 0.5). Move the threshold:

- **Lower the threshold** → flag more things positive → **recall ↑, precision ↓**
  (you catch more real cases but also raise more false alarms)
- **Raise the threshold** → flag fewer → **precision ↑, recall ↓**

So "improving the model" often means **choosing the threshold that matches your cost
trade-off**, not just training a better model. This is a classic interview talking point.

---

## 5. F1 score (malignant) — `0.9500`

**Formula:** harmonic mean = `2 · (precision · recall) / (precision + recall)`
**Plain meaning:** a single number balancing precision and recall.

**Why harmonic (not plain) mean:** it punishes imbalance. If precision is 1.0 but recall
is 0.0, the average is 0.5 but **F1 is 0** — you can't game it by maxing one and ignoring
the other. Use F1 when you need one headline number and both error types matter. If they
*don't* matter equally, use **Fβ** (β>1 weights recall, β<1 weights precision).

---

## 6. ROC-AUC — `0.9954`

**What ROC is:** sweep the threshold from 0→1 and plot **True Positive Rate (recall)**
against **False Positive Rate** at every setting. **AUC** = area under that curve.

**Plain meaning:** the probability that the model ranks a random positive *above* a random
negative. `0.5` = coin flip (useless), `1.0` = perfect ranking.

**Why it's valuable:** it's **threshold-independent** — it measures how well the model
*separates* the classes regardless of where you draw the line. So it's the best single
metric for comparing models before you've committed to an operating threshold.

**Its blind spot:** ROC-AUC can look great on heavily imbalanced data because the FP rate
has a huge denominator. When positives are rare, prefer **PR-AUC** (precision-recall AUC),
which focuses on the positive class.

---

## 7. ⚠️ Which class is "positive" here? (a real subtlety)

In scikit-learn's Breast Cancer dataset the labels are `0 = malignant`, `1 = benign`. The
*first* version of this project used the default `pos_label=1`, so precision/recall were
measured for detecting **benign** cases — which made the model look safer than it is.

For a *medical* problem that's the wrong framing: the costly mistake is calling a
**malignant** tumor benign (a **false negative on cancer** → a missed diagnosis). So
`train.py` now **relabels malignant as the positive class (1)** and reports recall on
malignant — because here a miss can be fatal, while a false alarm just means a follow-up
test. This is exactly the "the metric depends on what a mistake costs" point in action.

> **✅ Done — implemented in `train.py`.** The run now relabels malignant = positive and logs
> a full **confusion matrix**, a **per-class report**, **ROC + PR curves**, and a **threshold
> sweep** as MLflow artifacts. The payoff: it revealed the model misses 4/42 cancers at the
> default threshold — invisible under the old framing.

---

## 8. So what should *this* project optimize?

| If the goal is… | Optimize | Why |
|---|---|---|
| A medical screening tool | **Recall on malignant** | Missing a cancer (FN) is the catastrophic error |
| A balanced demo / model comparison | **ROC-AUC**, then **F1** | Threshold-independent separation, then a balanced headline |
| Minimizing unnecessary follow-up tests | **Precision on malignant** | Fewer false alarms |

Our baseline ranks classes almost perfectly (ROC-AUC 0.995), so it's a strong starting
point. The *next* question a mid-level engineer asks isn't "how do I raise accuracy" — it's
"**given the cost of a missed cancer, where do I set the threshold, and what recall does that
buy me?**"

**The actual threshold sweep (logged to `reports/threshold_sweep.txt`):**

| threshold | precision | recall | f1 | read it as |
|---|---|---|---|---|
| 0.30 | 0.9512 | 0.9286 | 0.9398 | catches more cancer, first false alarms appear |
| 0.40 | 0.9750 | 0.9286 | 0.9512 | **good screening operating point** |
| 0.50 (default) | 1.0000 | 0.9048 | 0.9500 | zero false alarms, but misses 4 cancers |
| 0.60 | 1.0000 | 0.8810 | 0.9367 | too conservative for a screen |
| 0.70 | 1.0000 | 0.8333 | 0.9091 | misses 7 cancers |

Moving 0.50 → **0.40** catches one more cancer (recall 0.905 → 0.929) for only a small
precision cost (1.000 → 0.975). For a screening tool that's the right trade.

---

## 9. What's implemented (and what's still open)

✅ Already in `src/train.py`:
- `confusion_matrix` — raw TP/FP/FN/TN, logged as counts **and** a plot
- `classification_report` — precision/recall/F1 **per class**, logged as a text artifact
- malignant set as the positive class (`pos_label`)
- **ROC curve** + **precision-recall curve** logged as MLflow artifacts
- a **threshold sweep** logging recall/precision/F1 at 0.3–0.7

🔜 Natural future additions:
- pick the operating threshold programmatically (e.g. "lowest threshold with precision ≥ 0.95")
  and **persist it with the model** so serving uses it
- cross-validation instead of a single split, for a confidence interval on each metric
- calibration curve (are the predicted probabilities trustworthy?)

---

## Reusing this for later phases

The same discipline carries into **Phase 2 (RAG)** and **Phase 3 (fine-tuning)** — those just
use *different* metrics (retrieval: precision@k / recall@k / MRR / nDCG; generation:
faithfulness, answer relevance). When we build those, we'll document each the same way:
**what it measures, when it matters, what a mistake costs.** See [[ML-skill-building-plan]].
