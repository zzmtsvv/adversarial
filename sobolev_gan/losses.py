from typing import Tuple
import torch
from torch.nn.modules.loss import _Loss
from torch.autograd import grad


class SobolevDiscriminatorLoss(_Loss):
    '''
        alpha - Lagrange multiplier
        rho - quadratic weight penalty
    '''
    def __init__(self,
                 alpha: float = 0.0,
                 rho: float = 1e-5,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

        self.alpha = alpha
        self.rho = rho
    
    def forward(self,
                real_image: torch.Tensor,
                generator_image: torch.Tensor,
                real_scores: torch.Tensor,
                generator_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(generator_scores.shape) == len(real_scores.shape) == 2  # [batch_size, 1]
        assert real_image.requires_grad == generator_image.requires_grad == True

        ipm_estimate = real_scores.mean() - generator_scores.mean()

        grad_real = grad(real_scores.sum(), real_image, create_graph=True)[0]
        grad_fake = grad(generator_scores.sum(), generator_image, create_graph=True)[0]
        
        grad_real = grad_real.view(grad_real.size(0), -1).pow(2).mean()
        grad_fake = grad_fake.view(grad_fake.size(0), -1).pow(2).mean()

        omega = (grad_real + grad_fake) / 2

        loss = -ipm_estimate - self.alpha * (1.0 - omega) + self.rho * (1.0 - omega).pow(2) / 2
        return ipm_estimate, loss


class SobolevGeneratorLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(self, generator_scores: torch.Tensor) -> torch.Tensor:
        assert len(generator_scores.shape) == 2  # [batch_size, 1]
        assert generator_scores.requires_grad

        loss = -generator_scores.mean()
        return loss
