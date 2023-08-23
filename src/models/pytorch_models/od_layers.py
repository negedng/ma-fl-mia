# https://github.com/SamsungLabs/ordered_dropout/blob/master/od/layers/cnn.py

import numpy as np
from torch import nn

__all__ = ["ODConv1d", "ODConv2d", "ODConv3d", "ODLinear"]


def od_conv_forward(layer, x, p=None):
    in_dim = x.size(1)  # second dimension is input dimension
    if not p:  # i.e., don't apply OD
        out_dim = layer.out_channels
    else:
        out_dim = int(np.ceil(layer.out_channels * p))
    # subsampled weights and bias
    weights_red = layer.weight[:out_dim, :in_dim]
    bias_red = layer.bias[:out_dim] if layer.bias is not None else None
    return layer._conv_forward(x, weights_red, bias_red)


class ODConv1d(nn.Conv1d):
    def __init__(self, p=None, *args, **kwargs):
        super(ODConv1d, self).__init__(*args, **kwargs)
        self.p = p

    def forward(self, x):
        return od_conv_forward(self, x, self.p)


class ODConv2d(nn.Conv2d):
    def __init__(self, p=None, *args, **kwargs):
        super(ODConv2d, self).__init__(*args, **kwargs)
        self.p = p

    def forward(self, x, p=None):
        return od_conv_forward(self, x, self.p)


class ODConv3d(nn.Conv3d):
    def __init__(self, p=None, *args, **kwargs):
        super(ODConv3d, self).__init__(*args, **kwargs)
        self.p = p

    def forward(self, x, p=None):
        return od_conv_forward(self, x, self.p)
    


class ODLinear(nn.Linear):
    def __init__(self, p=None, *args, **kwargs):
        super(ODLinear, self).__init__(*args, **kwargs)
        self.p = p

    def forward(self, x):
        in_dim = x.size(1)  # second dimension is input dimension
        p = self.p
        if not p:  # i.e., don't apply OD
            out_dim = self.out_features
        else:
            out_dim = int(np.ceil(self.out_features * p))
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return nn.functional.linear(x, weights_red, bias_red)


def sample(conf):
    if type(conf["scale_mode"]) == int and conf["scale_mode"] > 0 and conf["scale_mode"]<=conf["num_clients"]:
        model_size = np.random.choice(a=[0.5,1.0],
                                      size=1,
                                      p=[1-conf["scale_mode"]/conf["num_clients"],conf["scale_mode"]/conf["num_clients"]])
        return model_size
    m = 5
    return (np.random.randint(m) + 1) / m    