"""The candidate-inflation objective, and projected gradient ascent on it.

The adversary does not try to make the detector wrong. It maximises how many
candidates survive the confidence threshold, because that count is what sets the
suppression stage's work. Written against any callable that returns per-anchor class
scores, so it is not tied to one detector or one runtime.
"""
from __future__ import annotations

__all__ = ["soft_candidate_count", "pgd_inflate"]


def soft_candidate_count(scores, tau: float, temperature: float = 0.01):
    """Differentiable stand-in for "how many anchors clear ``tau``".

    ``scores`` is (anchors, classes) of per-class probabilities. The hard count is a
    step function with no useful gradient, so each anchor contributes
    ``sigmoid((max_c p - tau) / T)``: near 1 when it clears the threshold, near 0 when
    it does not, and steep in between. ``T = 0.01`` keeps it close to the step it
    replaces.

    The maximum over classes is taken FIRST, so this counts ANCHORS rather than
    class-expanded boxes. Counting after class expansion inflates the objective by
    roughly the class count and optimises a quantity the suppression stage never sees.
    """
    import torch
    per_anchor = scores.max(dim=-1).values
    return torch.sigmoid((per_anchor - tau) / temperature).sum()


def pgd_inflate(image, score_fn, tau: float, epsilon: float, step: float = 0.005,
                steps: int = 100, temperature: float = 0.01, *,
                random_start: bool = True, generator=None):
    """Maximise the surrogate count inside an L-inf ball. Returns the attacked image.

    ``image``         (1, C, H, W) float tensor in [0, 1]
    ``score_fn``      image -> (anchors, classes) per-class probabilities, differentiable
    ``random_start``  draw the first iterate uniformly in the epsilon-ball (default, and
                      what the paper's attack does); ``False`` starts at zero, as 2.1.0 did
    ``generator``     optional ``torch.Generator`` for the start; without one the global
                      generator is used, so ``torch.manual_seed`` makes the run repeatable

    The paper's attack starts from a point drawn uniformly in the epsilon-ball (its
    crafting run seeded the generator once, with 1337) and then takes 100 sign-gradient
    steps of 0.005. Up to 2.1.0 this function started at zero, which is a different
    attack: use ``random_start=False`` only to reproduce those versions. With a random
    start the result depends on the seed, so fix it and report it. Every iterate,
    including the first, is projected onto the epsilon-ball and onto the valid pixel
    range, so the returned image is one a client could actually submit.
    """
    import torch
    if random_start:
        start = torch.empty_like(image)
        start.uniform_(-epsilon, epsilon, generator=generator)
        delta = ((image + start).clamp(0.0, 1.0) - image).requires_grad_(True)
    else:
        delta = torch.zeros_like(image, requires_grad=True)
    for _ in range(steps):
        loss = soft_candidate_count(score_fn(image + delta), tau, temperature)
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta += step * grad.sign()                 # ASCENT: more candidates
            delta.clamp_(-epsilon, epsilon)
            delta.copy_((image + delta).clamp(0.0, 1.0) - image)
        delta.requires_grad_(True)
    return (image + delta).detach()
