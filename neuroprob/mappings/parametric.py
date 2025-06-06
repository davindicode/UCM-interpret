import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import numbers

from .. import base

        


### Generalized linear models ###
class GLM(base._input_mapping):
    """
    GLM rate model.
    """
    def __init__(self, input_dim, out_dims, w_len, bias=False, tensor_type=torch.float, 
                 active_dims=None):
        """
        :param int input_dims: total number of active input dimensions
        :param int out_dims: number of output dimensions
        :param int w_len: number of dimensions for the weights
        """
        super().__init__(input_dim, out_dims, tensor_type, active_dims)
        
        self.register_parameter('w', Parameter(torch.zeros((out_dims, w_len), dtype=self.tensor_type)))
        if bias:
            self.register_parameter('bias', Parameter(torch.zeros((out_dims), dtype=self.tensor_type)))
        else:
            self.bias = 0
            
            
    def set_params(self, w=None, bias=None):
        if w is not None:
            self.w.data = w.type(self.tensor_type).to(self.dummy.device)
        if bias is not None:
            self.bias.data = bias.type(self.tensor_type).to(self.dummy.device)

            
    def compute_F(self, XZ):
        """
        Default linear mapping
        
        :param torch.Tensor cov: covariates with shape (samples, timesteps, dims)
        :returns: inner product with shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        XZ = self._XZ(XZ)
        return (XZ*self.w[None, :, None, :]).sum(-1) + self.bias[None, :, None], 0
    
    
    def sample_F(self, XZ):
        return self.compute_F(XZ)[0]
    
    
    
class Bayesian_GLM(base._input_mapping):
    """
    GLM rate model.
    """
    def __init__(self, input_dim, out_dims, w_len, bias=False, tensor_type=torch.float, 
                 active_dims=None):
        """
        :param int input_dims: total number of active input dimensions
        :param int out_dims: number of output dimensions
        :param int w_len: number of dimensions for the weights
        """
        super().__init__(input_dim, out_dims, tensor_type, active_dims)
        
        self.register_parameter('qw_mu', Parameter(torch.zeros((out_dims, w_len), dtype=self.tensor_type)))
        self.register_parameter('qw_std', Parameter(torch.zeros((out_dims, w_len), dtype=self.tensor_type)))
        self.register_buffer('pw_mu', Parameter(torch.zeros((out_dims, w_len), dtype=self.tensor_type)))
        self.register_parameter('pw_std', Parameter(torch.zeros((out_dims, w_len), dtype=self.tensor_type)))
        if bias:
            self.register_parameter('bias', Parameter(torch.zeros((out_dims), dtype=self.tensor_type)))
        else:
            self.bias = 0
            
            
    def KL_prior(self, importance_weighted):
        """
        Ignores neuron, computes over all the output dimensions
        Note self.Luu is computed in compute_F or sample_F called before
        """
        if self.induc_pts.u_scale_tril is None: # log p(u)
            zero_loc = self.induc_pts.Xu.new_zeros(self.induc_pts.u_loc.shape)
            if self.whiten:
                identity = eye_like(self.induc_pts.Xu, zero_loc.shape[1])[None, ...].repeat(zero_loc.shape[0], 1, 1)
                p = distributions.Rn_MVN(zero_loc, scale_tril=identity)
            else: # loc (N, N_u), cov (N, N_u, N_u)
                p = distributions.Rn_MVN(zero_loc, scale_tril=self.Luu)
                
            return -p.log_prob(self.induc_pts.u_loc).sum()
        
        else: # log p(u)/q(u)
            zero_loc = self.induc_pts.u_loc.new_zeros(self.induc_pts.u_loc.shape)
            if self.whiten:
                identity = eye_like(self.induc_pts.u_loc, zero_loc.shape[1])[None, ...].repeat(zero_loc.shape[0], 1, 1)
                p = distributions.Rn_MVN(zero_loc, scale_tril=identity)#.to_event(zero_loc.dim() - 1)
            else: # loc (N, N_u), cov (N, N_u, N_u)
                p = distributions.Rn_MVN(zero_loc, scale_tril=self.Luu)#.to_event(zero_loc.dim() - 1)

            q = distributions.Rn_MVN(self.induc_pts.u_loc, scale_tril=self.induc_pts.u_scale_tril)#.to_event(self.u_loc.dim()-1)
            kl = torch.distributions.kl.kl_divergence(q, p).sum() # sum over neurons
            if torch.isnan(kl).any():
                kl = 0.
                print('Warning: sparse GP prior is NaN, ignoring prior term.')
            return kl
            
            
    def set_params(self, w=None, bias=None):
        if w is not None:
            self.w.data = w.type(self.tensor_type).to(self.dummy.device)
        if bias is not None:
            self.bias.data = bias.type(self.tensor_type).to(self.dummy.device)

            
    def compute_F(self, XZ):
        """
        Default linear mapping
        
        :param torch.Tensor cov: covariates with shape (samples, timesteps, dims)
        :returns: inner product with shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        XZ = self._XZ(XZ)
        return (XZ*self.w[None, :, None, :]).sum(-1) + self.bias[None, :, None], 0
    
    
    def sample_F(self, XZ):
        return# self.compute_F(XZ)[0]
    



### ANN ###
class FFNN(base._input_mapping):
    """
    Artificial neural network rate model.
    """
    def __init__(self, input_dim, out_dims, mu_ANN, sigma_ANN=None, tensor_type=torch.float, 
                 active_dims=None):
        """
        :param nn.Module mu_ANN: ANN parameterizing the mean function mapping
        :param nn.Module sigma_ANN: ANN paramterizing the standard deviation mapping if stochastic
        """
        super().__init__(input_dim, out_dims, tensor_type, active_dims)
        
        self.add_module('mu_ANN', mu_ANN)
        if sigma_ANN is not None:
            self.add_module('sigma_ANN', sigma_ANN)
        else:
            self.sigma_ANN = None
        
        
    def compute_F(self, XZ):
        """
        The input to the ANN will be of shape (samples*timesteps, dims).
        
        :param torch.Tensor cov: covariates with shape (samples, timesteps, dims)
        :returns: inner product with shape (samples, neurons, timesteps)
        :rtype: torch.tensor
        """
        XZ = self._XZ(XZ)
        incov = XZ.view(-1, XZ.shape[-1])
        post_mu = self.mu_ANN(incov).view(*XZ.shape[:2], -1).permute(0, 2, 1)
        if self.sigma_ANN is not None:
            post_var  = self.sigma_ANN(incov).view(*XZ.shape[:2], -1).permute(0, 2, 1)
        else:
            post_var = 0

        return post_mu, post_var

    
    def sample_F(self, XZ):
        self.compute_F(XZ)[0]