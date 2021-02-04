import torch
from toss.utils import listify


def compose(x, funcs, *args, order_key="_order", **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key):
        x = f(x, **kwargs)
    return x


def check_defaults(steppers, defaults):
    for stepper in steppers:
        for param, value in getattr(stepper, "DEFAULTS", {}).items():
            if param not in defaults:
                defaults[param] = value


class Optimizer:
    def __init__(self, params, steppers, **defaults):
        self.steppers = listify(steppers)
        check_defaults(self.steppers, defaults)

        self.param_groups = list(params)
        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]

        self.hypers = [{**defaults} for _ in self.param_groups]

    def grad_params(self):
        params = []
        for pg, hyper in zip(self.param_groups, self.hypers):
            for p in pg:
                if p.grad is not None:
                    params.append((p, hyper))
        return params

    def step(self):
        for p, hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def zero_grad(self):
        for p, hyper in self.grad_params():
            compose(p, self.steppers, **hyper)
