import torch


def handle_reduction(loss_func):
    def _inner(y, pred, reduction="mean"):
        loss = loss_func(y, pred)
        return loss.mean() if reduction == "mean" else loss.sum()
    return _inner

@handle_reduction
def MSE(y, pred, reduction="mean"):
    return (y-pred)**2

@handle_reduction
def MAE(y, pred, reduction="mean"):
    return torch.abs(y-pred)
