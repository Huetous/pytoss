import torch
from torch import nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BatchNorm(nn.Module):
    def __init__(self, n_filters, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.mults = nn.Parameter(torch.ones(n_filters, 1, 1))
        self.adds = nn.Parameter(torch.zeros(n_filters, 1, 1))
        self.register_buffer("vars", torch.ones(1, n_filters, 1, 1))
        self.register_buffer("means", torch.zeros(1, n_filters, 1, 1))

    def update_stats(self, x):
        m = x.mean((0, 2, 3), keepdim=True)
        v = x.var((0, 2, 3), keepdim=True)
        self.means.lerp_(m, self.momentum)
        self.vars.lerp_(v, self.momentum)
        return m, v

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                m, v = self.update_stats(x)
        else:
            m = self.means
            v = self.vars
        x = (x - m) / (v + self.eps).sqrt()
        return x * self.mults + self.adds


class GeneralReLU(nn.Module):
    def __init__(self, leak=None, sub=None, upper_bound=None, lower_bound=None):
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def forward(self, x):
        if self.leak is not None:
            x = F.leaky_relu(x, self.leak)
        else:
            F.relu(x)

        if self.sub is not None:
            x.sub_(self.sub)

        if self.upper_bound is not None:
            x.clamp_max_(self.upper_bound)

        if self.lower_bound is not None:
            x.clamp_min_(self.lower_bound)

        return x


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ConvLSUVLayer(nn.Module):
    def __init__(self, n_inputs, n_filters, ks=3, stride=2, sub=0., **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(n_inputs, n_filters, ks,
                              padding=ks // 2, stride=stride, bias=True)
        self.relu = GeneralReLU(sub=sub, **kwargs)
        self.bn = nn.BatchNorm2d(n_filters, eps=1e-5, momentum=0.1)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))

    @property
    def bias(self):
        return -self.relu.sub

    @bias.setter
    def bias(self, v):
        self.relu.sub = -v

    @property
    def weight(self):
        return self.conv.weight
