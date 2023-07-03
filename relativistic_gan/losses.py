import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class RelativisticDiscriminatorLoss(_Loss):
    # https://arxiv.org/abs/1807.00734v3
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_scores: torch.Tensor, generator_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(real_scores)
        labels_fake = torch.zeros_like(generator_scores)

        real_loss = self.loss(real_scores - generator_scores, labels_true)
        fake_loss = self.loss(generator_scores - real_loss, labels_fake)
        loss = (fake_loss + real_loss) / 2

        return loss


class RelativisticGeneratorLoss(_Loss):
    # https://arxiv.org/abs/1807.00734v3
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_scores: torch.Tensor, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == len(real_scores.shape) == 2  # [batch_size, 1]

        target = torch.ones_like(generator_scores)
        loss = self.loss(generator_scores - real_scores, target)
        return loss
