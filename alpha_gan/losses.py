import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class CodeReconstructionLoss(_Loss):
    '''
        the input is considered to be a hidden representation of the encoder
        and the desired latent distribution.
    '''
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.MSELoss()
    
    def forward(self, encoded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(encoded, target)


class DiscriminatorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.MSELoss()
    
    def forward(self, real_scores: torch.Tensor, generated_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(real_scores)
        labels_fake = torch.zeros_like(generated_scores)

        loss = self.loss(real_scores, labels_true) + self.loss(generated_scores, labels_fake)

        return loss


class GeneratorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.MSELoss()
    
    def forward(self, generator_score: torch.Tensor) -> torch.Tensor:
        assert len(generator_score.shape) == 2  # [batch_size, 1]

        target = torch.ones_like(generator_score)

        return self.loss(generator_score, target)


class ReconstructionLoss(_Loss):
    '''
        the input is considered to be real images and reconstructed ones by generator
        that takes hidden representation of these images from the encoder.
    '''
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.L1Loss()
    
    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        return self.loss(reconstructed, target)
