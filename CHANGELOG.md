# Changelog

## 2.1.0 (2026-10-30)

Correctness fixes to suppression; admission, bounds and selection are unchanged.

- `greedy_nms` visits candidates in the total order `(-score, index)` used by `admit_topk`.
  2.0.0 sorted with `scores.argsort()[::-1]`, which reversed the index order among tied
  scores, so the survivor of a group of tied overlapping boxes was the last index rather
  than the first. New keyword arguments, all optional: `classes` (explicit class-aware
  comparison), `limit` (stop after this many kept boxes, as a maximum detection count
  does) and `order_ids` (tie-break by global candidate index when suppressing an admitted
  subset). With the default arguments, results differ from 2.0.0 only where scores tie.
- Scores are validated and converted to float64 before ordering. Unsigned integer
  scores (negation wraps), the signed integer minimum (negation overflows) and boolean
  scores previously produced a wrong order or an error.
- `class_offset` raises when the stride does not exceed the span of the box coordinates.
  Detector boxes extend past the input canvas (-4.5 to 1296.9 px observed on a 1280-px
  input), so a stride of canvas size plus one let boxes of different classes at opposite
  corners suppress each other.
- Tests for the tie order, the prefix property (a capped run that keeps `limit` boxes
  returns the first `limit` boxes of the uncapped run, on quantized scores with many ties),
  integer and invalid scores, and the class stride.

## 2.0.0

Cluster-level Clopper-Pearson bound on the deadline-exceedance probability with a
three-valued verdict, and selection of K by fixed-sequence testing.

## 1.0.0

Initial release.
