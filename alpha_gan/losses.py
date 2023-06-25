import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class ReconstructionLoss(_Loss):
    '''
        the input is considered to be real images and reconstructed ones by generator
        that takes hidden representation of these images from the encoder.
    '''
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.L1Loss()
    
    def forward(self,
                reconstructed: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        return self.loss(reconstructed, target)


class EncoderLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, encoder_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(encoder_scores)
        labels_fake = torch.zeros_like(encoder_scores)

        loss = self.loss(encoder_scores, labels_true) - self.loss(encoder_scores, labels_fake)
        return loss


class GeneratorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                reconstructed_scores: torch.Tensor,
                generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == len(reconstructed_scores.shape) == 2  # [batch_size, 1]

        labels_true = torch.ones_like(reconstructed_scores)
        labels_fake = torch.zeros_like(generator_scores)

        recon_loss = self.loss(reconstructed_scores, labels_true) - self.loss(reconstructed_scores, labels_fake)
        gener_loss = self.loss(generator_scores, labels_true) - self.loss(generator_scores, labels_fake)
        return recon_loss + gener_loss


class DiscriminatorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                real_scores: torch.Tensor,
                reconstructed_scores: torch.Tensor,
                generated_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(real_scores)
        labels_fake = torch.zeros_like(generated_scores)

        loss = self.loss(real_scores, labels_true) + self.loss(generated_scores, labels_fake)
        loss += self.loss(reconstructed_scores, labels_fake)

        return loss


class CodecriminatorLoss(_Loss):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                encoder_scores: torch.Tensor,
                target_scores: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(encoder_scores)
        labels_fake = torch.zeros_like(target_scores)

        loss = self.loss(encoder_scores, labels_fake) + self.loss(target_scores, labels_true)
        return loss
