from toss.containers import ListContainer
from functools import partial
import torch
import matplotlib.pyplot as plt


class Hook:
    def __init__(self, layer, func):
        self.hook = layer.register_forward_hook(partial(func, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


class Hooks(ListContainer):
    def __init__(self, ms, f):
        super().__init__([Hook(m, f) for m in ms])

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self:
            h.remove()


def plot_hooks(hooks, size=None):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))
    for h in hooks:
        ms, ss, _ = h.stats
        size = size if size else 10
        ax0.plot(ms[:size])
        ax1.plot(ss[:size])
    plt.legend(range(len(hooks)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 6))
    for h in hooks:
        ms, ss, _ = h.stats
        ax0.plot(ms)
        ax1.plot(ss)
    plt.legend(range(len(hooks)))


def plot_hooks_act(hooks, first_n=4):
    def get_hist(h):
        return torch.stack(h.stats[2]).t().float().log1p()

    for h in hooks[:first_n]:
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.imshow(get_hist(h), origin="lower")
        ax.axis("off")
        plt.show()


def plot_hooks_min_act(hooks, first_n=4):
    def get_min(h):
        h1 = torch.stack(h.stats[2]).t().float()
        return h1[19:22].sum(0) / h1.sum(0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 6))
    for ax, h in zip(axes.flatten(), hooks[:first_n]):
        ax.plot(get_min(h))
        ax.set_ylim(0, 1)
    plt.tight_layout()