import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter




class Siren(nn.Module):
    """
    Activation function class for SIREN
    
    `Implicit Neural Representations with Periodic Activation Functions`,
    Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)
    
    
    
class Parallel_Linear(nn.Module):
    """
    Linear layers that separate different operations in one dimension.
    """
    __constants__ = ['in_features', 'out_features', 'channels']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, channels: int, bias: bool = True) -> None:
        """
        If channels is 1, then we share all the weights over the channel dimension.
        """
        super(Parallel_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        self.weight = Parameter(torch.Tensor(channels, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, channels: list) -> Tensor:
        """
        :param torch.tensor input: input of shape (batch, channels, in_dims)
        """
        if self.channels > 1: # separate weight matrices per channel
            W = self.weight.expand(1, self.channels, self.out_features, self.in_features)[:, channels, ...]
            B = 0 if self.bias is None else self.bias.expand(1, self.channels, self.out_features)[:, channels, :]
        else:
            W = self.weight[None, ...]
            B = self.bias[None, ...]
            
        return (W*input[..., None, :]).sum(-1) + B

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, channels={}, bias={}'.format(
            self.in_features, self.out_features, self.channels, self.bias is not None
        )
    
    
    
class Parallel_MLP(nn.Module):
    """
    Multi-layer perceptron class with parallel layers.
    """
    def __init__(self, layers, in_dims, out_dims, channels, nonlin=nn.ReLU(), 
                 out=None, bias=True, shared_W=False):
        super().__init__()
        self.in_dims = in_dims
        self.channels = channels
        c = 1 if shared_W else channels
        self.net = nn.ModuleList([])
        if len(layers) == 0:
            self.net.append(Parallel_Linear(in_dims, out_dims, c, bias=bias))
        else:
            self.net.append(Parallel_Linear(in_dims, layers[0], c, bias=bias))
            self.net.append(nonlin)
            for k in range(len(layers)-1):
                self.net.append(Parallel_Linear(layers[k], layers[k+1], c, bias=bias))
                self.net.append(nonlin)
            self.net.append(Parallel_Linear(layers[-1:][0], out_dims, c, bias=bias))
        if out is not None:
            self.net.append(out)
        
    def forward(self, input, channel_dims=None):
        """
        :param torch.tensor input: input of shape (samplesxtime, channelsxin_dims)
        """
        if channel_dims is None:
            channel_dims = list(range(self.channels))
        input = input.view(input.shape[0], -1, self.in_dims)
        
        for en, net_part in enumerate(self.net): # run through network list
            if en % 2 == 0:
                input = net_part(input, channel_dims)
            else:
                input = net_part(input)

        return input.view(input.shape[0], -1) # t, NxK


    
class MLP(nn.Module):
    """
    Multi-layer perceptron class
    """
    def __init__(self, layers, in_dims, out_dims, nonlin=nn.ReLU(), out=None, bias=True):
        super().__init__()
        net = nn.ModuleList([])
        if len(layers) == 0:
            net.append(nn.Linear(in_dims, out_dims, bias=bias))
        else:
            net.append(nn.Linear(in_dims, layers[0], bias=bias))
            net.append(nonlin)
            for k in range(len(layers)-1):
                net.append(nn.Linear(layers[k], layers[k+1], bias=bias))
                net.append(nonlin)
                #net.append(nn.BatchNorm1d())
            net.append(nn.Linear(layers[-1:][0], out_dims, bias=bias))
        if out is not None:
            net.append(out)
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        return self.net(input)
    
    
    
    
    
### networks ###
class BatchNorm(nn.Module):
    def __init__(self, eps=1.e-10):
        super().__init__()
        self.eps_cpu = torch.tensor(eps)
        self.register_buffer('eps', self.eps_cpu)

    def forward(self, z):
        """
        Do batch norm over batch and sample dimension
        """
        mean = torch.mean(z, dim=0, keepdims=True)
        std = torch.std(z, dim=0, keepdims=True)
        z_ = (z - mean) / (std + self.eps)
        return z_
    
    
    
class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, weight_norm=1.0, scale=False):
        super(WeightNormConv2d, self).__init__()
        
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_dim, out_dim, kernel_size, 
                stride=stride, padding=padding, bias=bias))
        if not scale:
            self.conv.weight_g.data = torch.ones_like(self.conv.weight_g.data)
            self.conv.weight_g.requires_grad = False

    def forward(self, x):
        return self.conv(x)
    
    
    
class WeightNormConvTranspose2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, weight_norm=1.0, scale=False):
        super(WeightNormConvTranspose2d, self).__init__()
        
        self.convT = nn.utils.weight_norm(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, 
                stride=stride, padding=padding, bias=bias))
        if not scale: # normalization is per filter
            self.convT.weight_g.data = torch.ones_like(self.convT.weight_g.data)
            self.convT.weight_g.requires_grad = False

    def forward(self, x):
        return self.convT(x)
    
    
    
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.tensor(mask.T).uint8())
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)
    