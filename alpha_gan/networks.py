# https://arxiv.org/abs/1706.04987
import torch
from torch import nn
from typing import List, Optional

try:
    from modules import GeneratorBlock, DiscriminatorBlock
    from utils import Reshape
except ModuleNotFoundError:
    from .modules import GeneratorBlock, DiscriminatorBlock
    from .utils import Reshape


class Encoder(nn.Module):
    '''
        Architecture is the same as in Discriminator except the head projects to the latent space with latent_dim
    '''
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 latent_dim: int = 128,
                 num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        first_dis_block = DiscriminatorBlock(in_channels, base_channels, downsample=True)
        first_dis_block.first_activation = nn.Identity()

        self.net = nn.Sequential(
            first_dis_block,
            self.make_layer(base_channels, 2 * base_channels, num_blocks[0]),
            self.make_layer(2 * base_channels, 4 * base_channels, num_blocks[1]),
            self.make_layer(4 * base_channels, 8 * base_channels, num_blocks[2]),
            self.make_layer(8 * base_channels, 16 * base_channels, num_blocks[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * base_channels, latent_dim)
        )

        self.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode='fan_in', nonlinearity="leaky_relu"
                )
    
    @staticmethod
    def make_layer(in_channels: int,
                   out_channels: int,
                   num_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            DiscriminatorBlock(in_channels, out_channels, downsample=True),
            *[DiscriminatorBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )
    
    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Generator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 latent_dim: int = 128,
                 num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256 * base_channels, bias=False),
            Reshape((-1, 16 * base_channels, 4, 4)),
            self.make_layer(16 * base_channels, 8 * base_channels, num_blocks[0]),
            self.make_layer(8 * base_channels, 4 * base_channels, num_blocks[1]),
            self.make_layer(4 * base_channels, 2 * base_channels, num_blocks[2]),
            self.make_layer(2 * base_channels, base_channels, num_blocks[3]),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, 1, 1, 0),
            nn.Tanh()
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode='fan_in', nonlinearity="leaky_relu"
                )
    
    @staticmethod
    def make_layer(in_channels: int,
                   out_channels: int,
                   num_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            GeneratorBlock(in_channels, out_channels),
            *[GeneratorBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out
    
    def sample(self, num_samples: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device

        if noise is None:
            noise = torch.randn((num_samples, self.latent_dim))
        
        return self.forward(noise.to(device))

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 num_blocks: List[int] = [1, 1, 1, 1]) -> None:
        super().__init__()

        first_dis_block = DiscriminatorBlock(in_channels, base_channels, downsample=True)
        first_dis_block.first_activation = nn.Identity()

        self.net = nn.Sequential(
            first_dis_block,
            self.make_layer(base_channels, 2 * base_channels, num_blocks[0]),
            self.make_layer(2 * base_channels, 4 * base_channels, num_blocks[1]),
            self.make_layer(4 * base_channels, 8 * base_channels, num_blocks[2]),
            self.make_layer(8 * base_channels, 16 * base_channels, num_blocks[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * base_channels, 1)
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode='fan_in', nonlinearity="leaky_relu"
                )

    @staticmethod
    def make_layer(in_channels: int,
                   out_channels: int,
                   num_blocks: int) -> nn.Sequential:
        return nn.Sequential(
            DiscriminatorBlock(in_channels, out_channels, downsample=True),
            *[DiscriminatorBlock(out_channels, out_channels) for _ in range(num_blocks - 1)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.net(x)
        return score
    
    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad = False

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
