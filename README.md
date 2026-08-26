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
result = greedy_nms(boxes[keep], scores[keep], iou_thres=0.65)
pairwise_bound(1024)                         # 523,776 comparisons, whatever the input
```

Choosing K from a deadline rather than from a round number:

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
| `nms.py` | greedy IoU suppression as the serving path runs it, and class offsetting |
| `attack.py` | the differentiable candidate-count surrogate and the PGD recurrence |
| `metrics.py` | tail quantile, deadline-miss ratio, session-clustered bootstrap interval |
| `selection.py` | the deadline-calibrated rule for choosing K |

## Three things that are easy to get wrong

**The tie rule is part of the specification.** `np.argpartition` is not stable, so when
more candidates tie at the cutoff than there are places left it returns an arbitrary
subset — and sorting the survivors afterwards settles their order while doing nothing
about which ones survived. Two runs can then admit different boxes and both claim to
"apply Top-K". Exact ties are reachable at fp16 and quantized scores. `admit_topk`
admits everything strictly above the cutoff and fills the rest by ascending index.

**Replayed requests are not independent trials.** The same image seen twice contributes
the same difficulty twice. `session_bootstrap_ci` resamples sessions and then
observations within them; pooling everything and resampling flat returns an interval
far too narrow, and an exact binomial limit on a miss ratio has the same defect.

**A bounded suppression stage is not a bounded service.** `T_NMS ≤ D_supp` does not
imply `L_e2e ≤ D_e2e`. Decode, inference, queueing, transport and serialisation are
outside this bound, and in the paper's matched-format experiment a clean request with
no adversary at all still missed the deadline on the decode term. The cap is necessary
for deadline integrity, not sufficient.

## Reported results

Measured on a deployed YOLOv8x-P2 serving path at 1280 px, against a 33.3 ms budget:
attacked suppression p95 falls from **672.42 ms uncapped to 4.28 ms at K = 1024**, for
a clean cost of **0.00064 AP@[.50:.95]** as a paired difference at the deployed
operating point. The bound holds on two detector geometries differing fourfold in
anchor count, on three suppression backends, on a second image corpus and on a
thermally throttled edge board. See the paper for the measurement scopes, the
intervals and the negative end-to-end result.

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
