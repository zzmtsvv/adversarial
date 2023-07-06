import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class DiscriminatorLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self,
                target_scores: torch.Tensor,
                encoder_scores: torch.Tensor) -> torch.Tensor:
        
        labels_true = torch.ones_like(target_scores)
        labels_fake = torch.zeros_like(encoder_scores)

        loss = self.loss(target_scores, labels_true) + self.loss(encoder_scores, labels_fake)
        return loss / 2


class GeneratorLoss(_Loss):
    def __init__(self,
                 adversarial_term: float,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pixelwise_loss = nn.L1Loss()
        self.alpha = adversarial_term
    
    def forward(self,
                encoder_scores: torch.Tensor,
                decoded_images: torch.Tensor,
                real_images: torch.Tensor) -> torch.Tensor:
        labels_true = torch.ones_like(encoder_scores)

        adversarial_loss = self.adversarial_loss(encoder_scores, labels_true)
        reconstruction_term = self.pixelwise_loss(decoded_images, real_images)

        loss = self.alpha * adversarial_loss + (1 - self.alpha) * reconstruction_term
        return loss
