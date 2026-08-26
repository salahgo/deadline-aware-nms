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
                steps: int = 100, temperature: float = 0.01):
    """Maximise the surrogate count inside an L-inf ball. Returns the attacked image.

    ``image``     (1, C, H, W) float tensor in [0, 1]
    ``score_fn``  image -> (anchors, classes) per-class probabilities, differentiable

    Deterministic: ``delta`` starts at zero and every step is the sign of the gradient,
    so there is no attack-seed variance to report and a replicate count is a timing
    session rather than a fresh attack. The iterate is projected back onto the
    epsilon-ball and onto the valid pixel range after every step, so the returned image
    is one a client could actually submit.
    """
    import torch
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
