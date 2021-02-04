import torch
from torch import tensor
from toss.utils import listify
import math
from functools import partial


def annealer(f):
    def _inner(start, end):
        return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start, end, pos):
    return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos):
    return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_exp(start, end, pos):
    return start * (end / start) ** pos


# sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])
def combine_scheds(ranges, scheds):
    assert sum(ranges) == 1.
    ranges = tensor([0] + listify(ranges))
    assert torch.all(ranges >= 0)
    ranges = torch.cumsum(ranges, 0)

    def _inner(pos):
        idx = (pos >= ranges).nonzero().max()
        actual_pos = (pos - ranges[idx]) / (ranges[idx + 1] - ranges[idx])
        return scheds[idx](actual_pos)

    return _inner
