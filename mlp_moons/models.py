"""模型模块：一个可配置的 MLP（隐藏层数、宽度、激活函数、Dropout 都可调）。"""

from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class MLP(nn.Module):
    """全连接前馈网络：in_dim -> [hidden]*n_hidden -> out_dim。

    Args:
        in_dim: 输入维度（make_moons 是 2）。
        hidden: 每个隐藏层的宽度。
        n_hidden: 隐藏层数量（>=1）。
        out_dim: 输出维度（二分类用 2 个 logits）。
        activation: 激活函数名，见 ACTIVATIONS。
        dropout: 隐藏层后的 dropout 概率，0 表示不用。
        batchnorm: 是否在激活前加 BatchNorm1d。
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden: int = 64,
        n_hidden: int = 2,
        out_dim: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if n_hidden < 1:
            raise ValueError("至少要有 1 个隐藏层")
        if activation not in ACTIVATIONS:
            raise ValueError(f"未知激活函数 {activation}，可选 {sorted(ACTIVATIONS)}")

        act_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden))
            if batchnorm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))  # 输出层不加激活，直接给 logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    """可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(**kwargs) -> MLP:
    return MLP(**kwargs)
