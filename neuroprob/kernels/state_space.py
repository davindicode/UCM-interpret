import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

from ..utils import linalg



class _state_space(nn.Module):
    """
    """
    def __init__(self, D, Q, tensor_type):
        super().__init__()
        self.tensor_type = tensor_type
        self.D = D # kernel dimensions
        self.Q = Q # state space dimensions
        
    def get_state_space(self):
        raise NotImplementedError
        
    def compute_covariance(self, F, Qc, L):
        """
        Default numerical computation
        Note Q = P0 - A P0 A
        """
        P0 = linalg.solve_lyapunov(F, -L @ L.T * Qc)
        return P0
        
    def compute_transition(self, dt, F):
        """
        Default numerical computation
        """
        return linalg.matrix_exponential(dt*F)

        
        
        
### LEG ###
class LEG(_state_space):
    """
    """
    def __init__(self, D, Q, D_N, out_dims, tensor_type=torch.float):
        """
        out_dims interpreted as mcxtrialsxN_copies
        """
        super().__init__(D, Q, tensor_type)
        self.out_dims = out_dims
        self.N = Parameter(torch.zeros((out_dims, Q, D_N), dtype=tensor_type))
        self.R = Parameter(torch.zeros((out_dims, Q, Q), dtype=tensor_type))
        self.H = Parameter(torch.ones((out_dims, D, Q), dtype=tensor_type))
        
    def get_state_space(self):
        N = self.N
        R = self.R
        
        F = -.5 * ((N[..., None, :] * N[..., None, :, :]).sum(-1) + R - R.transpose(-2, -1))
        
        Qc = 1.
        L = N
        H = self.H
        
        return F, Qc, L, H
    
    
    
class Bayesian_LEG(_state_space):
    """
    """
    def __init__(self, D, Q, D_N, out_dims, tensor_type=torch.float):
        """
        out_dims interpreted as mcxtrialsxN_copies
        """
        super().__init__(D, Q, tensor_type)
        self.out_dims = out_dims
        self.qmu_N = Parameter(torch.zeros((out_dims, Q, D_N), dtype=tensor_type))
        self.qmu_R = Parameter(torch.zeros((out_dims, Q, Q), dtype=tensor_type))
        self.qmu_H = Parameter(torch.ones((out_dims, D, Q), dtype=tensor_type))
        
        self.qstd_N = Parameter(torch.zeros((out_dims, Q, D_N), dtype=tensor_type))
        self.qstd_R = Parameter(torch.zeros((out_dims, Q, Q), dtype=tensor_type))
        self.qstd_H = Parameter(torch.ones((out_dims, D, Q), dtype=tensor_type))
        
        self.pstd_N = Parameter(torch.zeros((out_dims, Q, D_N), dtype=tensor_type))
        self.pstd_R = Parameter(torch.zeros((out_dims, Q, Q), dtype=tensor_type))
        self.pstd_H = Parameter(torch.ones((out_dims, D, Q), dtype=tensor_type))
        
    def get_state_space(self):
        N = self.N
        R = self.R
        
        F = -.5 * ((N[..., None, :] * N[..., None, :, :]).sum(-1) + R - R.permute(-2, -1))
        
        Qc = 1.
        L = N
        H = self.H
        
        return F, Qc, L, H
        
    
        


### kernels ###
class Lengthscale(_state_space):
    """
    """
    def __init__(self, D, Q, variance, lengthscale, f, tensor_type=torch.float):
        super().__init__(D, Q, tensor_type)
        
        if f == 'exp':
            self.lf = lambda x : torch.exp(x)
            self.lf_inv = lambda x: torch.log(x)
        elif f == 'softplus':
            self.lf = lambda x : F.softplus(x)
            self.lf_inv = lambda x: torch.where(x > 30, x, torch.log(torch.exp(x) - 1))
        elif f == 'relu':
            self.lf = lambda x : torch.clamp(x, min=0)
            self.lf_inv = lambda x: x
        else:
            raise NotImplementedError("Link function is not supported.")
            
        self.out_dims = lengthscale.shape[0]
        self._lengthscale = Parameter(self.lf_inv(lengthscale.type(tensor_type))) # N
        self._variance = Parameter(self.lf_inv(variance.type(tensor_type)))
    
    @property
    def variance(self):
        return self.lf(self._variance)[:, None] # out, T
    
    @variance.setter
    def variance(self):
        self._variance.data = self.lf_inv(variance)
        
    @property
    def lengthscale(self):
        return self.lf(self._lengthscale)[:, None] # out, T
    
    @lengthscale.setter
    def lengthscale(self):
        self._lengthscale.data = self.lf_inv(lengthscale)
    
    
    
class Exponential(Lengthscale):
    """
    Matern 1/2
    """
    def __init__(self, variance, lengthscale, tensor_type=torch.float):
        super().__init__(1, 1, variance, lengthscale, tensor_type)
        
    def get_state_space(self):
        Q = self.Q
        D = self.D
        
        lamb = 1./self.lengthscale
        
        F = torch.zeros((out_dims, Q, Q), dtype=self.tensor_type, device=self.H.device)
        F[:, 0, 0] = -lamb
        
        
        
        Qc = 2*lamb*self.variance
        L = torch.ones((out_dims, Q, Q), dtype=self.tensor_type, device=self.H.device)
        
        H = torch.ones((1, D, Q), dtype=self.tensor_type)
        return F, Qc, L, H
    
    def compute_covariance(self, F, Qc, L):
        """
        Analytical
        """
        P0 = self.variance[:, None, None]
        #*torch.ones((out_dims, Q, Q), dtype=self.tensor_type, device=self.H.device)
        return P0
        
    def compute_transition(self, dt, F):
        """
        Analytical
        """
        A = torch.exp(-dt/self.lengthscale)[:, None, None]
        #Q = P0 - A P0 A
        return A
        
        
        
    
class Matern32(Lengthscale):
    """
    Matern 3/2
    """
    def __init__(self, variance, lengthscale, tensor_type=torch.float):
        super().__init__(1, 2)
        
    def get_state_space(self):
        """
        F = torch.tensor(
            [[0., 1.], 
             [-lamb**2, -2*lamb]]
        )
        """
        Q = self.Q
        D = self.D
        lamb = np.sqrt(3.)/self.lengthscale
        
        
        F = torch.zeros((out_dims, Q, Q), dtype=self.tensor_type, device=self.H.device)
        F[:, 0, 0] = 0.
        F[:, 0, 1] = 1.
        F[:, 1, 0] = -lamb**2
        F[:, 1, 1] = -2*lamb
        
        Qc = 4*lamb**3*sigma**2
        L = torch.zeros((out_dims, Q, Q), dtype=self.tensor_type, device=self.H.device)
        L[:, 1, 1] = 1.
        
        H = torch.zeros((1, D, Q), dtype=self.tensor_type)
        H[0, 0, 1] = 1.
        
        return F, Qc, L, H
    
    def compute_covariance(self, F, Qc, L):
        """
        """
        P0 = torch.zeros((out_dims, Q, Q), dtype=self.tensor_type, device=self.H.device)
        P0[:, 0, 0] = self.variance
        P0[:, 1, 1] = 0
        return P0
    
    def compute_transition(self, dt, F):
        """
        Analytical
        """
        lamb = np.sqrt(3.)/self.lengthscale
        A = np.exp(-dt * lamb) * (dt * np.array([[lamb, 1.0], [-lamb**2.0, -lamb]]) + np.eye(2))
        return A
    


class Matern52(Lengthscale):
    """
    Matern 5/2
    """
    def __init__(self, variance, lengthscale, tensor_type=torch.float):
        super().__init__()
        
        
    def get_state_space(self):
        """
        F = torch.tensor(
            [[0.0, 1.0, 0.0], 
             [0., 0.0, 1.], 
             [-lamb**3, -3*lamb**2, -3*lamb]]
        )
        """
        Q = self.Q
        D = self.D
        lamb = np.sqrt(5.)/self.lengthscale
        
        Qc = 16*lamb**5*sigma**2/3
        L = torch.tensor(
            [[0., 0., 0.], 
             [0., 0., 0.], 
             [0., 0., 1.]]
        ).float()
        
        return F, Qc, L, H
        
    def compute_covariance(self, F, Qc, L):
        """
        """
        kappa = 5.0 / 3.0 * var / ell**2.0
        Pinf = np.array([[var,    0.0,   -kappa],
                         [0.0,    kappa, 0.0],
                         [-kappa, 0.0,   25.0*var / ell**4.0]])
        
        return P0
    
    def compute_transition(self, dt):
        lam = np.sqrt(5.0) / ell
        dtlam = dt * lam
        A = np.exp(-dtlam) \
            * (dt * np.array([[lam * (0.5 * dtlam + 1.0),      dtlam + 1.0,            0.5 * dt],
                              [-0.5 * dtlam * lam ** 2,        lam * (1.0 - dtlam),    1.0 - 0.5 * dtlam],
                              [lam ** 3 * (0.5 * dtlam - 1.0), lam ** 2 * (dtlam - 3), lam * (0.5 * dtlam - 2.0)]])
               + np.eye(3))
        return A



class Matern72(Lengthscale):
    """
    Matern 7/2
    """
    def __init__(self, variance, lengthscale, tensor_type=torch.float):
        super().__init__()
        
        
    def get_state_space(self):
        """
        F = torch.tensor(
            [[0., 1., .0, .0], 
             [0., .0, 1., .0], 
             [0., .0, 0., 1.], 
             [-lamb**4, -4*lamb**3, -6*lamb**2, -4*lamb]]
        )
        """
        Q = self.Q
        D = self.D
        lamb = np.sqrt(7.)/self.lengthscale
        Qc = 10976/7**3/5*sigma**2*lamb**7
        L = torch.tensor(
            [[0., 0., 0., 0.], 
             [0., 0., 0., 0.], 
             [0., 0., 0., 0.], 
             [0., 0., 0., 1.]]
        ).float()
        
        return F, P0, Qc, L, H
        
    def compute_covariance(self, F, Qc, L):
        """
        """
        kappa = 7.0 / 5.0 * var / ell**2.0
        kappa2 = 9.8 * var / ell**4.0
        Pinf = np.array([[var,    0.0,     -kappa, 0.0],
                         [0.0,    kappa,   0.0,    -kappa2],
                         [-kappa, 0.0,     kappa2, 0.0],
                         [0.0,    -kappa2, 0.0,    343.0*var / ell**6.0]])
        
        
        
        return P0
    
    def compute_transition(self, dt):
        lam = np.sqrt(7.0) / ell
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam ** 2
        A = np.exp(-dtlam) \
            * (dt * np.array([[lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),      1.0 + dtlam + 0.5 * dtlam2,
                              0.5 * dt * (1.0 + dtlam),                       dt ** 2 / 6],
                              [-dtlam2 * lam ** 2.0 / 6.0,                    lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                              1.0 + dtlam - 0.5 * dtlam2,                     dt * (0.5 - dtlam / 6.0)],
                              [lam3 * dtlam * (dtlam / 6.0 - 0.5),            dtlam * lam2 * (0.5 * dtlam - 2.0),
                              lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),       1.0 - dtlam + dtlam2 / 6.0],
                              [lam2 ** 2 * (dtlam - 1.0 - dtlam2 / 6.0),      lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                              lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),      lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0)]])
               + np.eye(4))
        return A