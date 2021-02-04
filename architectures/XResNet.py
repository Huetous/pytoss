import torch.nn as nn
from toss.layers import Flatten
from toss.architectures.cnn_utils import conv_layer, init_cnn


def noop(x):
    return x


def make_layer(expansion, n_in, n_out, n_blocks, stride):
    return nn.Sequential(
        *[ResBlock(expansion, n_in if i == 0 else n_out, n_out, stride if i == 0 else 1)
          for i in range(n_blocks)])


class ResBlock(nn.Module):
    def __init__(self, expansion, n_in, n_hid, stride=1):
        super().__init__()
        n_out, ni = n_hid * expansion, n_in * expansion
        layers = [conv_layer(ni, n_hid, 1)]

        if expansion == 1:
            layers += [conv_layer(n_hid, n_out, 3, stride=stride,
                                  zero_bn=True, act=False)]
        else:
            layers += [
                conv_layer(n_hid, n_hid, 3, stride=stride),
                conv_layer(n_hid, n_out, 1, zero_bn=True, act=False)
            ]

        self.left_path = nn.Sequential(*layers)
        self.right_path = noop if n_in == n_out else conv_layer(ni, n_out, 1, act=False)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.left_path(x) + self.right_path(self.pool(x)))


class XResNet(nn.Module):
    def __init__(self, expansion, blocks, channels_in=3, channels_out=10):
        super().__init__()
        n_filters = [channels_in, (channels_in + 1) * 8, 64, 64]
        stem = [conv_layer(n_filters[i], n_filters[i + 1], stride=2 if i == 0 else 1)
                for i in range(3)]

        n_filters = [64 // expansion, 64, 128, 256, 512]
        stages = [make_layer(expansion, n_filters[i], n_filters[i + 1],
                             n_blocks=n_blocks, stride=1 if i == 0 else 2)
                  for i, n_blocks in enumerate(blocks)]

        self.model = nn.Sequential(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *stages,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(n_filters[-1] * expansion, channels_out))

        init_cnn(self.model)

    def forward(self, x):
        return self.model(x)
