import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class DiscriminatorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.MSELoss()
    
    def forward(self, real_scores: torch.Tensor, generated_scores: torch.Tensor) -> torch.Tensor:
        loss = 0

        labels_true = torch.ones_like(real_scores)
        labels_fake = torch.zeros_like(generated_scores)

        loss += self.loss(real_scores, labels_true) + self.loss(generated_scores, labels_fake)

        return loss


class GeneratorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.base_loss = nn.MSELoss()

    def forward(self, generator_score: torch.Tensor) -> torch.Tensor:
        assert len(generator_score.shape) == 2  # [batch_size, 1]

        target = torch.ones_like(generator_score)
        loss = self.base_loss(generator_score, target)
        return loss
