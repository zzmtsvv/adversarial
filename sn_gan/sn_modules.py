from typing import Union, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from numpy.linalg import svd


DEFAULT_DTYPE = torch.float32


def l2_normalize(x: torch.Tensor, eps=1e-12):
    return x / (x.pow(2).sum() + eps).sqrt()


class SNLinear(nn.Module):
    dtype = DEFAULT_DTYPE

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 use_gamma: bool = True,
                 pow_iter=1,
                 lip_const=1):
        super(SNLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_gamma = use_gamma
        self.pow_iter = pow_iter
        self.lip_const = lip_const

        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=self.dtype))

        if bias:
            self.register_parameter(
                "bias", nn.Parameter(torch.zeros((out_features, ), dtype=self.dtype)))
        else:
            self.register_buffer("bias", None)

        self.lip_const = lip_const
        self.register_buffer("u", torch.randn((out_features, ), dtype=self.dtype))

        if use_gamma:
            self.register_parameter(
                "gamma", nn.Parameter(torch.ones((1, ), dtype=self.dtype)))
        else:
            self.register_parameter("gamma", None)

        # initialize the parameters
        nn.init.kaiming_normal_(self.weight, a=0., mode="fan_in")

    def _init_gamma(self):
        if self.use_gamma:
            nn.init.constant_(
                self.gamma, svd(self.weight.data, compute_uv=False)[0])

    @property
    def weight_bar(self) -> torch.Tensor:
        sigma, u = self.power_iteration(self.weight, self.u, self.pow_iter)
        if self.training:
            self.u = u
        weight_bar = self.lip_const * self.weight / sigma
        if self.use_gamma:
            weight_bar = self.gamma * weight_bar
        return weight_bar
    
    @staticmethod
    def power_iteration(w: torch.Tensor, u_init: torch.Tensor, num_iter=1) -> Tuple[torch.Tensor, torch.Tensor]:
        # w: (F_out, F_in)
        # u_init: (F_out, )
        u = u_init
        with torch.no_grad():
            for _ in range(num_iter - 1):
                v = l2_normalize(u @ w)
                u = l2_normalize(w @ v)

            v = l2_normalize(u @ w)

        wv = w @ v  # node allows gradient flow
        u = l2_normalize(wv.detach())
        sigma = u @ wv

        return sigma, u

    def forward(self, x):
        return F.linear(x, self.weight_bar, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SNConv2d(nn.Module):
    dtype = DEFAULT_DTYPE

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: int,
                 padding: int,
                 bias: bool = True,
                 use_gamma: bool = True,
                 pow_iter=1,
                 lip_const=1):
        super(SNConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_gamma = use_gamma
        self.pow_iter = pow_iter
        self.lip_const = lip_const

        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels, kernel_size, kernel_size), dtype=self.dtype))
        
        if bias:
            self.register_parameter(
                "bias", nn.Parameter(torch.zeros((out_channels, ), dtype=self.dtype)))
        else:
            self.register_buffer("bias", None)
        
        self.lip_const = lip_const
        self.register_buffer("u", torch.randn((out_channels, ), dtype=self.dtype))
        
        if use_gamma:
            self.register_parameter(
                "gamma", nn.Parameter(torch.ones((1, ), dtype=self.dtype)))
        else:
            self.register_buffer("gamma", None)

        nn.init.kaiming_normal_(self.weight, a=0., mode="fan_in")

    def _init_gamma(self):
        if self.use_gamma:
            nn.init.constant_(
                self.gamma, svd(
                    self.weight.data.reshape(self.out_channels, -1), compute_uv=False)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight_bar, self.bias, stride=self.stride, padding=self.padding)

    @property
    def weight_bar(self) -> torch.Tensor:
        sigma, u = self.power_iteration(self.weight, self.u, self.pow_iter)

        if self.training:
            self.u = u

        weight_bar = self.lip_const * self.weight / sigma
        if self.use_gamma:
            weight_bar = self.gamma * weight_bar
        
        return weight_bar
    
    @staticmethod
    def power_iteration(w, u_init, num_iter=1) -> Tuple[torch.Tensor, torch.Tensor]:
        # w: (c_out, C_in, K, K)
        # u_init: (c_out, )
        u = u_init
        with torch.no_grad():
            for _ in range(num_iter - 1):
                v = l2_normalize(torch.einsum("i,ijkl->jkl", u, w))
                u = l2_normalize(torch.einsum("ijkl,jkl->i", w, v))

            v = l2_normalize(torch.einsum("i,ijkl->jkl", u, w))

        wv = torch.einsum("ijkl,jkl->i", w, v)  # node allows gradient flow
        u = l2_normalize(wv.detach())
        sigma = (u * wv).sum()

        return sigma, u

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
