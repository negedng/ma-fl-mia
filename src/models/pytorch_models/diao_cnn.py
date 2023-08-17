import numpy as np
import torch
import torch.nn as nn
from src.models.pytorch_models.layer_utils import get_norm, get_scaler, get_conv, get_linear


class DiaoCNN(nn.Module):
    """Model following the diao et al paper.
    Emmiting LN, GN and IN as it is not straightforward to cast to TF,
    and the paper shows superiority of the BN

    https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/tree/master
    """

    def __init__(
        self,
        input_shape=(3, 32, 32),
        num_classes=10,
        model_rate=1.0,
        static_bn=False,
        use_scaler=True,
        keep_scaling=False,
        norm_mode="bn",
        default_hidden=[64, 128, 256, 512],
        use_bias=True,
        ordered_dropout=False
    ):
        super(DiaoCNN, self).__init__()

        hidden_sizes = [int(np.ceil(model_rate * x)) for x in default_hidden]
        # hidden_sizes = [int(np.ceil(model_rate * x)) if i==2 else x for i,x in enumerate(default_hidden)]
        scaler_rate = model_rate

        self.od_layers = []
        conv = get_conv(in_channels=input_shape[0], out_channels=hidden_sizes[0], kernel_size=3, stride=1, padding=1, bias=use_bias, ordered_dropout=ordered_dropout, p=1.0)
        self.od_layers.append(conv)

        norm = get_norm(hidden_sizes[0], norm_mode=norm_mode, static_bn=static_bn)
        scaler = get_scaler(
            use_scaler=use_scaler, scaler_rate=scaler_rate, keep_scaling=keep_scaling
        )

        blocks = [
            conv,
            scaler,
            #norm,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]

        for i in range(len(hidden_sizes) - 1):
            conv = get_conv(in_channels=hidden_sizes[i], out_channels=hidden_sizes[i + 1], kernel_size=3, stride=1, padding=1, bias=use_bias, ordered_dropout=ordered_dropout, p=1.0)
            self.od_layers.append(conv)

            norm = get_norm(
                hidden_sizes[i + 1], norm_mode=norm_mode, static_bn=static_bn
            )
            scaler = get_scaler(
                use_scaler=use_scaler,
                scaler_rate=scaler_rate,
                keep_scaling=keep_scaling,
            )
            blocks.extend(
                [
                    conv,
                    scaler,
                    #norm,
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        blocks = blocks[:-1]
        linear = get_linear(in_features=hidden_sizes[-1], out_features=num_classes, bias=use_bias, ordered_dropout=ordered_dropout, p=1.0)

        blocks.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                linear,
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.blocks(x)
        return output
    
    def set_ordered_dropout_rate(self, p=None):
        for layer in self.od_layers:
            layer.p = p

def get_diao_CNN(*args, **kwargs):
    return DiaoCNN(*args, **kwargs)
