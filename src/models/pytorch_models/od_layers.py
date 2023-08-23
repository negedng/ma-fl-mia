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


class ODBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, p=None, *args, **kwargs):
        super(ODBatchNorm2d, self).__init__(*args, **kwargs)
        self.p = p
    
    def forward(self, x):
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # OD part
        out_dim = int(np.ceil(self.num_features * self.p))
        weights_red = self.weight[:out_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None

        return nn.functional.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weights_red,
            bias_red,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

class ODGroupNorm(nn.GroupNorm):
    def __init__(self, p=None, reduce_groups=False, *args, **kwargs):
        super(ODGroupNorm, self).__init__(*args, **kwargs)
        self.p = p
        self.reduce_groups = reduce_groups
    
    def forward(self, x):
        num_groups_red = int(np.ceil(self.p * self.num_groups)) if self.reduce_groups else self.num_groups
        # OD part
        out_dim = int(np.ceil(self.num_channels * self.p))
        weights_red = self.weight[:out_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return nn.functional.group_norm(
            x,
            num_groups_red,
            weights_red,
            bias_red,
            self.eps
        )

def sample(conf):
    if type(conf["scale_mode"]) == int and conf["scale_mode"] > 0 and conf["scale_mode"]<=conf["num_clients"]:
        model_size = np.random.choice(a=[0.5,1.0],
                                      size=1,
                                      p=[1-conf["scale_mode"]/conf["num_clients"],conf["scale_mode"]/conf["num_clients"]])
        return model_size
    m = 5
    return (np.random.randint(m) + 1) / m    