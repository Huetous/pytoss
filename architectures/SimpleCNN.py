import torch
import torch.nn as nn
from toss.architectures.cnn_utils import conv_layer, init_cnn
from toss.layers import Flatten


class SimpleCNN(nn.Module):
    def __init__(self, channels_in, channels_out, n_filters, include_head=True):
        super().__init__()
        n_filters = [channels_in] + n_filters
        layers = [conv_layer(n_filters[i], n_filters[i + 1], stride=2)
                  for i in range(len(n_filters) - 1)
                  ] + [nn.AdaptiveAvgPool2d(1), Flatten()]
        if include_head:
            layers += [nn.Linear(n_filters[-1], channels_out)]

        self.model = nn.Sequential(*layers)
        init_cnn(self.model)

    def forward(self, x):
        return self.model(x)


