import numpy as np
import torch
import torch.nn as nn
from src.models.pytorch_models.layer_utils import get_norm, get_scaler


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
    ):
        super(DiaoCNN, self).__init__()

        hidden_sizes = [int(np.ceil(model_rate * x)) for x in default_hidden]
        # hidden_sizes = [int(np.ceil(model_rate * x)) if i==2 else x for i,x in enumerate(default_hidden)]
        scaler_rate = model_rate

        norm = get_norm(hidden_sizes[0], norm_mode=norm_mode, static_bn=static_bn)
        scaler = get_scaler(
            use_scaler=use_scaler, scaler_rate=scaler_rate, keep_scaling=keep_scaling
        )

        blocks = [
            nn.Conv2d(input_shape[0], hidden_sizes[0], 3, 1, 1),
            scaler,
            norm,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]

        for i in range(len(hidden_sizes) - 1):
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
                    nn.Conv2d(hidden_sizes[i], hidden_sizes[i + 1], 3, 1, 1),
                    scaler,
                    norm,
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        blocks = blocks[:-1]
        blocks.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden_sizes[-1], num_classes),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.blocks(x)
        return output

def get_diao_CNN(*args, **kwargs):
    return DiaoCNN(*args, **kwargs)
