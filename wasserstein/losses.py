import torch
from torch.nn.modules.loss import _Loss


class WassersteinDiscriminatorLoss(_Loss):
    # https://arxiv.org/abs/1701.07875
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, real_scores: torch.Tensor, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == len(real_scores.shape) == 2  # [batch_size, 1]

        loss = -torch.mean(real_scores) + torch.mean(generator_scores)
        return loss


class WassersteinGeneratorLoss(_Loss):
    # https://arxiv.org/abs/1701.07875
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == 2  # [batch_size, 1]

        loss = -torch.mean(generator_scores)
        return loss
