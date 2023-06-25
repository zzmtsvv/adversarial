import torch
from torch import nn
from typing import Tuple, List, Optional


class GeneratorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out += self.skip_connection(x)
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 downsample: bool = False) -> None:
        super().__init__()

        self.first_activation = nn.ReLU()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1 + downsample, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels == out_channels and not downsample:
            self.skip_connection = nn.Identity()
        else:
            skip = [nn.AvgPool2d(2)] if downsample else []
            skip.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
            self.skip_connection = nn.Sequential(*skip)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(self.first_activation(x))
        out += self.skip_connection(x)
        return out
