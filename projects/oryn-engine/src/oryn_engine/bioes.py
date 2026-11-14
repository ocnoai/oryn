"""Bounded constrained BIOES decoding for native token-classification heads."""

from __future__ import annotations


def transition_mask(types):
    import torch

    count = 1 + 4 * len(types)
    mask = torch.zeros(count, count, dtype=torch.bool)
    starts, ends = [0], [0]
    for offset in range(1, count, 4):
        begin, inside, end, single = range(offset, offset + 4)
        starts.extend((begin, single))
        ends.extend((end, single))
        for source in (begin, inside):
            mask[source, inside] = mask[source, end] = True
    for source in ends:
        mask[source, starts] = True
    return mask


def decode(logits, mask, begin_bias, end_bias):
    """Return the best complete path; ties use the first native label index."""
    import torch

    logits = logits.detach().float().cpu()
    if (
        logits.ndim != 2
        or not 1 <= logits.shape[0] <= 4096
        or not 5 <= logits.shape[1] <= 257
        or (logits.shape[1] - 1) % 4
        or not bool(torch.isfinite(logits).all())
        or mask.dtype != torch.bool
        or tuple(mask.shape) != (logits.shape[1],) * 2
    ):
        raise ValueError("BIOES_DECODER_GEOMETRY")
    count = logits.shape[1]
    transitions = torch.full((count, count), float("-inf"))
    transitions[mask.cpu()] = 0.0
    transitions[:, 1::4] += float(begin_bias)
    transitions[3::4, :] += float(end_bias)
    if not torch.isfinite(transitions[mask.cpu()]).all():
        raise ValueError("BIOES_DECODER_BIAS")
    score = logits[0].clone()
    score[2::4] = score[3::4] = float("-inf")
    pointers = []
    for emission in logits[1:]:
        score, previous = (score[:, None] + transitions).max(dim=0)
        score = score + emission
        pointers.append(previous)
    score[1::4] = score[2::4] = float("-inf")
    path = [int(score.argmax())]
    for previous in reversed(pointers):
        path.append(int(previous[path[-1]]))
    return list(reversed(path))
