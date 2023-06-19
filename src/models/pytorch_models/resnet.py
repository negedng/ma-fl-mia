import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pytorch_models.layer_utils import get_norm, get_scaler

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, scaler_rate, static_bn, use_scaler, keep_scaling, norm_mode):
        super(Block, self).__init__()
        n1 = get_norm(in_planes, norm_mode=norm_mode, static_bn=static_bn)
        n2 = get_norm(planes, norm_mode=norm_mode, static_bn=static_bn)
        self.scaler = get_scaler(
            use_scaler=use_scaler, scaler_rate=scaler_rate, keep_scaling=keep_scaling
        )

        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(self.scaler(x)))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(self.scaler(out))))
        out += shortcut
        return out




class ResNet(nn.Module):
    def __init__(self, 
                 input_shape=(3,32,32),
                 default_hidden=[64, 128, 256, 512],
                 block=Block, num_blocks=[2, 2, 2, 2],
                 num_classes=10,
                 model_rate=1.0,
                 static_bn=False,
                 use_scaler=True,
                 keep_scaling=False,
                 norm_mode="bn"):
        super(ResNet, self).__init__()
        hidden_size = [int(np.ceil(model_rate * x)) for x in default_hidden]
        scaler_rate = model_rate
        
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(input_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1, scaler_rate=scaler_rate, static_bn=static_bn, use_scaler=use_scaler, keep_scaling=keep_scaling, norm_mode=norm_mode)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2, scaler_rate=scaler_rate, static_bn=static_bn, use_scaler=use_scaler, keep_scaling=keep_scaling, norm_mode=norm_mode)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2, scaler_rate=scaler_rate, static_bn=static_bn, use_scaler=use_scaler, keep_scaling=keep_scaling, norm_mode=norm_mode)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2, scaler_rate=scaler_rate, static_bn=static_bn, use_scaler=use_scaler, keep_scaling=keep_scaling, norm_mode=norm_mode)
        self.n4 = get_norm(hidden_size[3] * block.expansion, norm_mode=norm_mode, static_bn=static_bn)
        self.scaler = get_scaler(
            use_scaler=use_scaler, scaler_rate=scaler_rate, keep_scaling=keep_scaling
        )
        self.linear = nn.Linear(hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, scaler_rate, static_bn, use_scaler, keep_scaling, norm_mode):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, scaler_rate, static_bn, use_scaler, keep_scaling, norm_mode))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.n4(self.scaler(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

