import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
from numbers import Number

from .. import distributions, base




class histogram(base._input_mapping):
    """
    Histogram rate model based on GLM framework.
    Has an identity link function with positivity constraint on the weights.
    Only supports regressor mode.
    """
    def __init__(self, bins_cov, out_dims, ini_rate=1.0, alpha=None, tensor_type=torch.float, active_dims=None):
        """
        The initial rate should not be zero, as that gives no gradient in the Poisson 
        likelihood case
        
        :param tuple bins_cov: tuple of bin objects (np.linspace)
        :param int neurons: number of neurons in total
        :param float ini_rate: initial rate array
        :param float alpha: smoothness prior hyperparameter, None means no prior
        """
        super().__init__(len(bins_cov), out_dims, tensor_type, active_dims)
        ini = torch.tensor([ini_rate]).view(-1, *np.ones(len(bins_cov)).astype(int))
        self.register_parameter('w', Parameter(ini*torch.ones((out_dims,) + \
                                               tuple(len(bins)-1 for bins in bins_cov), 
                                                              dtype=self.tensor_type)))
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=self.tensor_type))
        else:
            self.alpha = alpha
        self.bins_cov = bins_cov # use PyTorch for integer indexing
        
        
    def set_params(self, w=None):
        if w is not None:
            self.w.data = torch.tensor(w, device=self.w.device, dtype=self.tensor_type)
        
        
    def compute_F(self, XZ):
        XZ = self._XZ(XZ)
        samples = XZ.shape[0]
        
        tg = []
        for k in range(self.input_dims):
            tg.append(torch.bucketize(XZ[..., k], self.bins_cov[k], right=True)-1)

        XZ_ind = (torch.arange(samples)[:, None, None].expand(-1, self.out_dims, len(tg[0][b])), 
                  torch.arange(self.out_dims)[None, :, None].expand(samples, -1, len(tg[0][b])),) + \
                  tuple(tg_[b][:, None, :].expand(samples, self.out_dims, -1) for tg_ in tg)
 
        return self.w[XZ_ind], 0


    def sample_F(self, XZ):
        return self.compute_F(XZ)[0]
    
    
    def KL_prior(self, importance_weighted):
        if self.alpha is None:
            return super().KL_prior(importance_weighted)
    
        smooth_prior = self.alpha[0]*(self.w[:, 1:, ...] - self.w[:, :-1, ...]).pow(2).sum() + \
            self.alpha[1]*(self.w[:, :, 1:, :] - self.w[:, :, :-1, :]).pow(2).sum() + \
            self.alpha[2]*(self.w[:, ..., 1:] - self.w[:, ..., :-1]).pow(2).sum()
        return -smooth_prior
    
    
    def constrain(self):
        self.w.data = torch.clamp(self.w.data, min=0)

        
    def set_unvisited_bins(self, ini_rate=1.0, unvis=np.nan):
        """
        Set the bins that have not been updated to unvisited value unvis.
        """
        self.w.data[self.w.data == ini_rate] = torch.tensor(unvis, device=self.w.device)
        
        

class spline(base._input_mapping):
    """
    Spline based mapping.
    """
    def __init__(self, bins_cov, out_dims, ini_rate=1.0, alpha=None, tensor_type=torch.float, active_dims=None):
        """
        The initial rate should not be zero, as that gives no gradient in the Poisson 
        likelihood case
        
        :param tuple bins_cov: tuple of bin objects (np.linspace)
        :param int neurons: number of neurons in total
        :param float ini_rate: initial rate array
        :param float alpha: smoothness prior hyperparameter, None means no prior
        """
        super().__init__(len(bins_cov), out_dims, tensor_type, active_dims)