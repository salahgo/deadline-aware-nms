# Deadline-Aware Hardening of Real-Time Object Detection Against Candidate-Inflation Latency Attacks

Reference implementation of the defense, the attack objective and the deadline
metrics from the paper of the same name.

An NMS-based detector performs an amount of work set by the number of candidates that
survive its confidence threshold — and that number is chosen by the input. A
candidate-inflation attack drives it far outside its clean range, and the deadline is
what pays: suppression alone can take hundreds of milliseconds on a request whose
budget is 33.3 ms. A faster suppression backend lowers the constant and leaves the
attacked count free.

This admits at most **K** candidates to NMS, which bounds the pairwise comparisons an
attacker can provoke. It needs no retraining, no weight modification and no change to
the detector — it acts at the exported runtime boundary, so it applies to an already
promoted artifact.

## Install

```bash
pip install -e .            # numpy only
pip install -e ".[attack]"  # adds torch, for the PGD attack objective
```

## Use

```python
import numpy as np
from deadline_nms import admit_topk, greedy_nms, pairwise_bound

keep   = admit_topk(scores, k=1024)          # deterministic even when the cutoff ties
result = keep[greedy_nms(boxes[keep], scores[keep], iou_thres=0.65,
                         classes=classes[keep], limit=300, order_ids=keep)]
pairwise_bound(1024)                         # 523,776 comparisons, whatever the input
```

Choosing K from a deadline rather than from a round number, with a certified bound
(v2.0.0): every independently sampled input contributes all its requests, and a grid point
is selected only if the bound PASSES. The grid is tested in increasing K and selection stops at
the first point that does not PASS (fixed-sequence testing), so the selected K keeps the 95%
level without a multiplicity correction:

```python
from deadline_nms import select_k_certified, exceedance_bound
k_star, per_k = select_k_certified(
    grid=[512, 1024, 2048, 4096, 5120],
    latencies=lat,                 # K -> {input id: [suppression ms of every request]}
    accuracy_cost=cost,            # K -> paired clean accuracy given up
    budget_ms=33.3, max_accuracy_cost=1e-3)
exceedance_bound(served_by_input, deadline_ms=33.3)   # PASS / FAIL / INCONCLUSIVE
```

The 1.x rule (a Student-t limit on the mean of session p95 values) is kept for
compatibility but deprecated: it is not a bound on the p95 of future requests.

```python
from deadline_nms import select_k

select_k(grid=[256, 512, 1024, 2048],
         suppression_sessions=tails,     # K -> per-session suppression p95, ms
         accuracy_cost=cost,             # K -> paired clean accuracy given up
         budget_ms=33.3,
         max_accuracy_cost=1e-3)
```

Run the mechanism end to end on synthetic candidates — no dataset, no weights, no GPU,
a few seconds:

```bash
PYTHONPATH=. python examples/demo.py
pip install pytest && PYTHONPATH=. python -m pytest tests -q
```

## What is here

| Module | Contents |
|---|---|
| `admission.py` | the cap, with membership defined lexicographically in `(-score, index)` |
| `nms.py` | greedy IoU suppression in the admission order `(-score, index)`, class-aware and with an output limit (v2.1.0); class offsetting |
| `attack.py` | the differentiable candidate-count surrogate and the PGD recurrence |
| `metrics.py` | tail quantile, deadline-miss ratio, session-clustered bootstrap interval (descriptive) |
| `bounds.py` | exact cluster-level bound on P(latency > deadline), three-valued verdict (v2.0.0) |
| `selection.py` | the deadline-calibrated rule for choosing K (`select_k_certified`, v2.0.0) |

## Four things that are easy to get wrong

**The tie rule is part of the specification.** `np.argpartition` is not stable, so when
more candidates tie at the cutoff than there are places left it returns an arbitrary
subset — and sorting the survivors afterwards settles their order while doing nothing
about which ones survived. Two runs can then admit different boxes and both claim to
"apply Top-K". Exact ties are reachable at fp16 and quantized scores. `admit_topk`
admits everything strictly above the cutoff and fills the rest by ascending index.

**Suppression must use the same order as admission.** Breaking score ties one way when
admitting and another way when suppressing makes the kept set depend on the sorting
algorithm. Since v2.1.0 `greedy_nms` visits candidates in `(-score, index)` order, the
key `admit_topk` uses; pass `order_ids=keep` when suppressing an admitted subset. With
one order, identical overlap arithmetic and the same output limit, a capped run that
already keeps `limit` boxes returns exactly the first `limit` boxes of the uncapped run
(tested). Use `classes=` rather than `class_offset` for class-aware suppression: an
offset stride must exceed the coordinate span, and detector boxes extend past the canvas.

**Replayed requests are not independent trials, and a deadline claim is about future
requests.** The same image seen twice contributes the same difficulty twice. Since
`Q_0.95(T) <= D` exactly when `P(T > D) <= 0.05`, `bounds.exceedance_bound` bounds that
probability over independently sampled INPUTS, each contributing all its requests. It is
valid for any dependence within an input, and it does not collapse to [0, 0] or [1, 1] when
every request meets or misses the deadline, as a bootstrap of miss indicators does. With
zero misses it needs at least 59 independent inputs to certify 0.05 at 95%;
`min_inputs_for_pass()` returns that number.

**A bounded suppression stage is not a bounded service.** `T_NMS ≤ D_supp` does not
imply `L_e2e ≤ D_e2e`. Decode, inference, queueing, transport and serialisation are
outside this bound, and in the paper's matched-format experiment a clean request with
no adversary at all still missed the deadline on the decode term. The cap is necessary
for deadline integrity, not sufficient.

## Reported results

Measured on a deployed YOLOv8x-P2 serving path at 1280 px, against a 33.3 ms budget, over
137 VisDrone inputs (one frame per video sequence) and five replay sessions: attacked
suppression p95 on the deployed backend falls from **557.8 ms uncapped to 3.91 ms at
K = 1024**, for a clean cost of **0.00064 AP@[.50:.95]** as a paired difference at the
deployed operating point. The bound holds on two detector geometries differing fourfold in
anchor count, on three suppression backends and on a second image corpus. On the edge
board suppression time scales with the clock, so K must be selected at the lowest clock the
platform reaches. Deadline verdicts are conditional on the runtime states measured; where a
worker's state persists for its lifetime, take the restart as the unit. See the paper for the
measurement scopes, the bounds and the end-to-end results, including the one configuration
that meets 30 Hz under attack.

## Changes

See [CHANGELOG.md](CHANGELOG.md).

The absolute milliseconds are properties of the platform they were taken on. The
portable result is the bound on admitted work.

## Citation

If you use this, please cite the paper:

```bibtex
@article{gontara_deadline_aware_nms,
  author  = {Gontara, Salah and Trabelsi, Selem and Ben Khalifa, Khaled},
  title   = {Deadline-Aware Hardening of Real-Time Object Detection Against
             Candidate-Inflation Latency Attacks},
  journal = {Journal of Real-Time Image Processing},
  year    = {2026}
}
```

The archived release of this repository has its own DOI; see `CITATION.cff`.

## License

MIT — see `LICENSE`.

## Changelog

**2.0.0** (2026-10, JRTIP revision). Added `bounds.py` (exact cluster-level Clopper-Pearson
bound on the deadline-exceedance probability, PASS/FAIL/INCONCLUSIVE verdict) and
`select_k_certified`, which tests the K grid in fixed sequence. `upper_confidence_limit` and
`select_k` are deprecated. The 1.x
t-limit bounded the mean of session p95 values, not the p95 of future requests, and the 1.x
release used t(0.95, n-1) while the paper's computation used t(0.975, n-1). In
`select_k_certified` a grid point without data is reported as MISSING and stops the sequence.
`exceedance_bound`'s docstring states the scope of its heterogeneous-input guarantee (the
verdicts, at the paper's sample sizes, for independent inputs). `examples/demo.py` uses the
certified selection.

**1.0.0** (2026-08). Initial release.
