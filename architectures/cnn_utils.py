import torch.nn as nn


def init_cnn(module):
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, 0)

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)

    for m in module.children():
        init_cnn(m)


def conv(n_in, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(n_in, nf, kernel_size=ks,
                     stride=stride, padding=ks // 2, bias=bias)


def conv_layer(n_in, n_out, kernel_size=3, stride=1, bn=True, zero_bn=False, act=True):
    layers = [conv(n_in, n_out, kernel_size, stride)]

    if bn:
        bn = nn.BatchNorm2d(n_out)
        nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
        layers.append(bn)

    if act:
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)
