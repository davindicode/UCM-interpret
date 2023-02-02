import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
from numbers import Number

from .. import distributions, base
from ..utils.signal import eye_like
from ..utils.linalg import PSD_solve




# linear algebra computations
#@torch.jit.script
def p_F_U(X, X_u, kernelobj, f_loc, f_scale_tril, full_cov=False,
          compute_cov=True, whiten=False, jitter=1e-6):
    r"""
    Computed Gaussian conditional distirbution.

    :param int out_dims: number of output dimensions
    :param torch.Tensor X: Input data to evaluate the posterior over
    :param torch.Tensor X_u: Input data to conditioned on, inducing points in sparse GP
    :param GP.kernels.Kernel kernel: A kernel module object
    :param torch.Tensor f_loc: Mean of :math:`q(f)`. In case ``f_scale_tril=None``,
        :math:`f_{loc} = f`
    :param torch.Tensor f_scale_tril: Lower triangular decomposition of covariance
        matrix of :math:`q(f)`'s
    :param torch.Tensor Lff: Lower triangular decomposition of :math:`kernel(X, X)`
        (optional)
    :param string cov_type: A flag to decide what form of covariance to compute
    :param bool whiten: A flag to tell if ``f_loc`` and ``f_scale_tril`` are
        already transformed by the inverse of ``Lff``
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition
    :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    N_u = X_u.size(-2) # number of inducing, inducing points
    T = X.size(2) # timesteps from KxNxTxD
    K = X.size(0)
    out_dims = f_loc.shape[0]
    
    Kff = kernelobj(X_u[None, ...])[0, ...].contiguous()
    Kff.data.view(Kff.shape[0], -1)[:, ::N_u+1] += jitter  # add jitter to diagonal
    Lff = torch.linalg.cholesky(Kff) # N, N_u, N_u
    
    Kfs = kernelobj(X_u[None, ...], X) # K, N, N_u, T

    N_l = Kfs.shape[1]
    Kfs = Kfs.permute(1, 2, 0, 3).reshape(N_l, N_u, -1) # N, N_u, KxT

    if N_l == 1: # single lengthscale for all outputs
        # convert f_loc_shape from N, N_u to 1, N_u, N
        f_loc = f_loc.permute(-1, 0)[None, ...]

        # f_scale_tril N, N_u, N_u
        if f_scale_tril is not None:
            # convert f_scale_tril_shape from N, N_u, N_u to N_u, N_u, N, convert to 1 x 2D tensor for packing
            f_scale_tril = f_scale_tril.permute(-2, -1, 0).reshape(1, N_u, -1)

    else: # multi-lengthscale
        # convert f_loc_shape to N, N_u, 1
        f_loc = f_loc[..., None]
        # f_scale_tril N, N_u, N_u

    if whiten:
        v_4D = f_loc[None, ...].repeat(K, 1, 1, 1) # K, N, N_u, N_
        W = torch.linalg.solve_triangular(Lff, Kfs, upper=False)
        W = W.view(N_l, N_u, K, T).permute(2, 0, 3, 1) # K, N, T, N_u
        if f_scale_tril is not None:
            S_4D = f_scale_tril[None, ...].repeat(K, 1, 1, 1)
            
    else:
        pack = torch.cat((f_loc, Kfs), dim=-1) # N, N_u, L
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril), dim=-1)

        Lffinv_pack = torch.linalg.solve_triangular(Lff, pack, upper=False)
        v_4D = Lffinv_pack[None, :, :, :f_loc.size(-1)].repeat(K, 1, 1, 1) # K, N, N_u, N_
        if f_scale_tril is not None:
            S_4D = Lffinv_pack[None, :, :, -f_scale_tril.size(-1):].repeat(K, 1, 1, 1) # K, N, N_u, N_u or N_xN_u

        W = Lffinv_pack[:, :, f_loc.size(-1):f_loc.size(-1)+K*T]
        W = W.view(N_l, N_u, K, T).permute(2, 0, 3, 1) # K, N, T, N_u

    if N_l == 1:
        loc = W.matmul(v_4D).permute(0, 3, 2, 1)[..., 0] # K, N, T
    else:
        loc = W.matmul(v_4D)[..., 0] # K, N, T
        
    if compute_cov is False: # e.g. kernel ridge regression
        return loc, 0, Lff
        
    
    if full_cov:
        Kss = kernelobj(X)
        Qss = W.matmul(W.transpose(-2, -1))
        cov = Kss - Qss # K, N, T, T
    else:
        Kssdiag = kernelobj(X, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        # due to numerical errors, clamp to avoid negative values
        cov = (Kssdiag - Qssdiag).clamp(min=0) # K, N, T

    if f_scale_tril is not None:
        W_S = W.matmul(S_4D) # K, N, T, N_xN_u
        if N_l == 1:
            W_S = W_S.view(K, 1, T, N_u, out_dims).permute(0, 4, 2, 3, 1)[..., 0]

        if full_cov:
            St_Wt = W_S.transpose(-2, -1)
            K = W_S.matmul(St_Wt)
            cov = cov + K
        else:
            Kdiag = W_S.pow(2).sum(dim=-1)
            cov = cov + Kdiag
    
    return loc, cov, Lff





#@torch.jit.script
def fixed_interval_Kalman_filter(A, H, P0, y_tilde, v_tilde):
    """
    filtering, parallelized over first tensor dimension (mc x trials x out_dims)
    """
    ss_d = A.shape[-1]
    f_d = H.shape[-2]

    # filter distributions
    timesteps = y_tilde.shape[1]
    m_f = torch.empty((y_tilde.shape[0], timesteps, ss_d), 
                    device=A.device, dtype=A.dtype)
    P_f = torch.empty((y_tilde.shape[0], timesteps, ss_d, ss_d), 
                    device=A.device, dtype=A.dtype)

    # initialize
    m_f[:, 0, :] = 3
    P_f[:, 0, :] = 3

    # loop
    log_Z = 0
    for step in range(timesteps):
        # prediction step
        m_p = (A * m_f[:, step:step+1, :]).sum(-1)
        P_p = P0 + ((A[..., None] * (P_f[:, step:step+1, None, ...] - P0)).sum(-2)[..., None] * \
                A.transpose(-2, -1)[..., None, :, :]).sum(-1)

        # compute innovation
        eta = y_tilde - H @ m_p
        s = H @ P_p @ H + v_tilde
        log_Z = log_Z + .5 * (torch.log(2*np.pi * s) + s**2/eta)

        k = P_p @ h / s
        m_f[:, step+1, :] = m_p + eta * k
        P_f[:, step+1, ...] = P_p - k (h * P_p)

    return m_f, P_f, log_Z




#@torch.jit.script
def fixed_interval_RTS_smoother(A, H, P0, m_f, P_f):
    """
    Fixed interval smoothing
    """
    ss_d = F.shape[-1]
    f_d = H.shape[-2]

    #if full_state:
    #    m_s = torch.empty((t.shape[0], timesteps, ss_d), 
    #                      device=self.dummy.device, dtype=self.tensor_type)
    #    P_s = torch.empty((t.shape[0], timesteps, ss_d, ss_d), 
    #                      device=self.dummy.device, dtype=self.tensor_type)
    #else:
    timesteps = m_f.shape[1]
    m_s = torch.empty((m_f.shape[0], timesteps, f_d), 
                      device=A.device, dtype=A.dtype)
    P_s = torch.empty((m_f.shape[0], timesteps, f_d, f_d), 
                      device=A.device, dtype=A.dtype)

    # initialize
    m_s[:, -1, :] = m_f[:, -1, :]
    P_s[:, -1, ...] = P_f[:, -1, ...]

    # loop
    for step in range(timesteps-2, -1, -1):
        Gt = PSD_solve(P_s, A * P_f[:, step, ...])
        G = Gt.transpose(-2, -1)

        # prediction step
        m_p = (A * m_f[:, step:step+1, :]).sum(-1)
        P_p = P0 + ((A[..., None] * (P_f[:, step:step+1, None, ...] - P0)).sum(-2)[..., None] * \
                A.transpose(-2, -1)[..., None, :, :]).sum(-1)

        m_s[:, step, :] = m_f[:, step, :] + G * (m_s[:, step+1, :] - m_p[:, step+1, :])
        P_s[:, step+1, ...] = P_f + G * (P_s[:, step+1, ...] - P_p[:, step+1, ...]) * Gt

        if full_state is False:
            m_s = H @ m_s
            P_s = H @ P_s @ H

    return m_s, P_s





# GP models
class _GP(base._input_mapping):
    """
    """
    def __init__(self, input_dims, out_dims, mean, learn_mean, tensor_type, active_dims):
        super().__init__(input_dims, out_dims, tensor_type, active_dims)
        
        ### GP mean ###
        if isinstance(mean, Number):
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean, dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean, dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, torch.Tensor):
            if mean.shape != torch.Size([out_dims]):
                raise ValueError('Mean dimensions do not match output dimensions')
            if learn_mean:
                self.mean = Parameter(mean[None, :, None].type(self.tensor_type))
            else:
                self.register_buffer('mean', mean[None, :, None].type(self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, nn.Module):
            self.add_module("mean_function", mean)
            if learn_mean is False:
                self.mean_function.requires_grad = False
                
        else:
            raise NotImplementedError('Mean type is not supported.')
            



class SVGP(_GP):
    """
    Sparse Variational Gaussian Process model with covariates for regression and latent variables.
    """
    def __init__(self, input_dims, out_dims, kernel, inducing_points, mean=0.0, learn_mean=False, 
                 MAP=False, whiten=False, kernel_regression=False, tensor_type=torch.float, jitter=1e-6, active_dims=None):
        r"""
        Specify the kernel type with corresponding dimensions and arguments.
        
        .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f),

        where :math:`p(y \mid f)` is the likelihood.

        We will use a variational approach in this model by approximating :math:`q(f,u)`
        to the posterior :math:`p(f,u \mid y)`. Precisely, :math:`q(f) = p(f\mid u)q(u)`,
        where :math:`q(u)` is a multivariate normal distribution with two parameters
        ``u_loc`` and ``u_scale_tril``, which will be learned during a variational
        inference process.

        The sparse model has :math:`\mathcal{O}(NM^2)` complexity for training,
        :math:`\mathcal{O}(M^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs, :math:`M` is the number of inducing inputs. Size of
        variational parameters is :math:`\mathcal{O}(M^2)`.
        
        
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param nn.Module inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        :param Kernel kernel: a tuple listing kernels, with content 
                                     (kernel_type, topology, lengthscale, variance)
        :param Number/torch.Tensor/nn.Module mean: initial GP mean of shape (samples, neurons, timesteps), or if nn.Module 
                                        learnable function to compute the mean given input
        """
        super().__init__(input_dims, out_dims, mean, learn_mean, tensor_type, active_dims)
        self.jitter = jitter
        self.whiten = whiten
        self.kernel_regression = kernel_regression
        
        ### kernel ###
        if kernel.input_dims != self.input_dims:
            ValueError('Kernel dimensions do not match expected input dimensions')
        if kernel.tensor_type != self.tensor_type:
            ValueError('Kernel tensor type does not match model tensor type')
        self.add_module("kernel", kernel)
        
        ### inducing points ###
        if inducing_points.input_dims != input_dims:
            raise ValueError('Inducing point dimensions do not match expected dimensions')
        if inducing_points.out_dims != out_dims:
            raise ValueError('Inducing variable output dimensions do not match model')
        if kernel.tensor_type != self.tensor_type:
            ValueError('Inducing point tensor type does not match model tensor type')
        self.add_module('induc_pts', inducing_points)
        self.Luu = None # needs to be computed in sample_F/compute_F
        
            
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
    
    
    def constrain(self):
        self.induc_pts.constrain()
            
        
    def compute_F(self, XZ):
        """
        Computes moments of marginals :math:`q(f_i|\bm{u})` and also updating :math:`L_{uu}` matrix
        model call uses :math:`L_{uu}` for the MVN, call after this function
        
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        
        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).
            
        .. note:: Variational parameters ``u_loc``, ``u_scale_tril``, the
            inducing-point parameter ``Xu``, together with kernel's parameters have
            been learned.
        
        covariance_type is a flag to decide if we want to predict full covariance matrix or 
        just variance.
        
        .. note:: The GP is centered around zero with fixed zero mean, but a learnable 
            mean is added after computing the posterior to get the mapping mean.
        
        XZ # K, N, T, D
        X_u = self.Xu[None, ...] # K, N, T, D
        
        :param X: input regressors with shape (samples, timesteps, dims)
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor), both of shape ()
        """
        XZ = self._XZ(XZ)
        loc, var, self.Luu = p_F_U(
            XZ, self.induc_pts.Xu, self.kernel, self.induc_pts.u_loc, self.induc_pts.u_scale_tril, 
            compute_cov=~self.kernel_regression, full_cov=False, whiten=self.whiten, jitter=self.jitter
        )
        
        return loc + self.mean_function(XZ), var
    
    
    def sample_F(self, XZ, samples=1, eps=None):
        """
        Samples from the variational posterior :math:`q(\bm{f}_*|\bm{u})`, which can be the predictive distribution
        """
        XZ = self._XZ(XZ)
        
        if XZ.shape[-1] < 1000: # sample naive way
            loc, cov, self.Luu = p_F_U(
                XZ, self.induc_pts.Xu, self.kernel, self.induc_pts.u_loc, self.induc_pts.u_scale_tril, 
                compute_cov=True, full_cov=True, whiten=self.whiten, jitter=self.jitter
            )
            cov.view(-1, cov.shape[-1]**2)[:, ::cov.shape[-1]+1] += self.jitter
            L = cov.double().cholesky().type(self.tensor_type)

            if samples > 1: # expand
                XZ = XZ.repeat(samples, 1, 1, 1)
                L = L.repeat(samples)

            if eps is None: # sample random vector
                eps = torch.randn(XZ.shape[:-1], dtype=self.tensor_type, device=cov.device)

            return loc + self.mean_function(XZ) + \
                   (L * eps[..., None, :]).sum(-1)
        
        else: # decoupled sampling
            eps_f, eps_u = eps
            
            return loc + self.mean_function(XZ) + \
                   (L * eps[..., None, :]).sum(-1)
        
    
    
    
class tSVGP(_GP):
    """
    Sparse Variational Gaussian Process model with covariates for regression and latent variables.
    """
    def __init__(self, input_dims, out_dims, kernel, inducing_points, mean=0.0, learn_mean=False, 
                 MAP=False, whiten=False, kernel_regression=False, tensor_type=torch.float, jitter=1e-6, active_dims=None):
        r"""
        Site-based approximation
        
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param nn.Module inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        :param Kernel kernel: a tuple listing kernels, with content 
                                     (kernel_type, topology, lengthscale, variance)
        :param Number/torch.Tensor/nn.Module mean: initial GP mean of shape (samples, neurons, timesteps), or if nn.Module 
                                        learnable function to compute the mean given input
        
        References:
        
        [1] `Dual Parameterization of Sparse Variational Gaussian Processes`, 
        Vincent Adam, Paul E. Chang, Mohammad Emtiyaz Khan, Arno Solin (2021)
        """
        super().__init__(input_dims, out_dims, mean, learn_mean, tensor_type, active_dims)
        
        self.register_buffer('lambda_tilde_1', lambda_tilde_1)
        self.register_buffer('lambda_tilde_2', lambda_tilde_2)
        
        ### kernel ###
        if kernel.input_dims != self.input_dims:
            ValueError('Kernel dimensions do not match expected input dimensions')
        if kernel.tensor_type != self.tensor_type:
            ValueError('Kernel tensor type does not match model tensor type')
        self.add_module("kernel", kernel)
        
        ### inducing points ###
        if inducing_points.input_dims != input_dims:
            raise ValueError('Inducing point dimensions do not match expected dimensions')
        if inducing_points.out_dims != out_dims:
            raise ValueError('Inducing variable output dimensions do not match model')
        if kernel.tensor_type != self.tensor_type:
            ValueError('Inducing point tensor type does not match model tensor type')
        self.add_module('induc_pts', inducing_points)
        self.Luu = None # needs to be computed in sample_F/compute_F
        
            
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
    
    
    def constrain(self):
        self.induc_pts.constrain()
            
        
    def compute_F(self, XZ):
        """
        Computes moments of marginals :math:`q(f_i|\bm{u})` and also updating :math:`L_{uu}` matrix
        model call uses :math:`L_{uu}` for the MVN, call after this function
        
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        
        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).
            
        .. note:: Variational parameters ``u_loc``, ``u_scale_tril``, the
            inducing-point parameter ``Xu``, together with kernel's parameters have
            been learned.
        
        covariance_type is a flag to decide if we want to predict full covariance matrix or 
        just variance.
        
        .. note:: The GP is centered around zero with fixed zero mean, but a learnable 
            mean is added after computing the posterior to get the mapping mean.
        
        XZ # K, N, T, D
        X_u = self.Xu[None, ...] # K, N, T, D
        
        :param X: input regressors with shape (samples, timesteps, dims)
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor), both of shape ()
        """
        XZ = self._XZ(XZ)
        loc, var, self.Luu = p_F_U(
            XZ, self.induc_pts.Xu, self.kernel, self.induc_pts.u_loc, self.induc_pts.u_scale_tril, 
            compute_cov=~self.kernel_regression, full_cov=False, whiten=self.whiten, jitter=self.jitter
        )
        
        return loc + self.mean_function(XZ), var
    
    
    def sample_F(self, XZ, samples=1, eps=None):
        """
        Samples from the variational posterior :math:`q(\bm{f}_*|\bm{u})`, which can be the predictive distribution
        """
        XZ = self._XZ(XZ)
        
        if XZ.shape[-1] < 1000: # sample naive way
            loc, cov, self.Luu = p_F_U(
                XZ, self.induc_pts.Xu, self.kernel, self.induc_pts.u_loc, self.induc_pts.u_scale_tril, 
                compute_cov=True, full_cov=True, whiten=self.whiten, jitter=self.jitter
            )
            cov.view(-1, cov.shape[-1]**2)[:, ::cov.shape[-1]+1] += self.jitter
            L = cov.double().cholesky().type(self.tensor_type)

            if samples > 1: # expand
                XZ = XZ.repeat(samples, 1, 1, 1)
                L = L.repeat(samples)

            if eps is None: # sample random vector
                eps = torch.randn(XZ.shape[:-1], dtype=self.tensor_type, device=cov.device)

            return loc + self.mean_function(XZ) + \
                   (L * eps[..., None, :]).sum(-1)
        
        else: # decoupled sampling
            eps_f, eps_u = eps
            
            return loc + self.mean_function(XZ) + \
                   (L * eps[..., None, :]).sum(-1)
        
    
    

    
class CVI_VSSGP(_GP):
    """
    Variational Gaussian Process model for temporal regression
    No minibatching
    """
    def __init__(self, input_dims, out_dims, state_space, Tu, learn_Tu=False, uniform_Tu=False, 
                 mean=0.0, learn_mean=False, tensor_type=torch.float, jitter=1e-6, active_dims=None):
        r"""
        Using conjugate computation variational inference and state space formulation of GPs [1].
        One can evaluate the posterior at values away from inducing points using doubly sparse structure [2].
        
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param torch.Tensor inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        
        References:
        
        [1] `FAST VARIATIONAL LEARNING IN STATE-SPACE GAUSSIAN PROCESS MODELS`, 
        Paul E. Chang, William J. Wilkinson, Mohammad Emtiyaz Khan, Arno Solin (2020)
        
        [2] `Doubly Sparse Variational Gaussian Processes`, 
        Vincent Adam, Stefanos Eleftheriadis, Nicolas Durrande, Artem Artemev, James Hensman (2020)
        """
        super().__init__(input_dims, out_dims, mean, learn_mean, tensor_type, active_dims)
        
        ### kernel ###
        self.add_module("state_space", state_space)
        
        ### State space GP setup ###
        if learn_Tu:
            self.Tu = Parameter(Tu.type(self.tensor_type))
            if uniform_Tu:
                raise ValueError('Cannot have uniformly separated time points while learning time locations')
            self.dt = None # need to compute
            self.uniform_Tu = False
            
        else:
            self.register_buffer('Tu', Tu.type(self.tensor_type))
            self.register_buffer('dt', self.Tu[:, 1:]-self.Tu[:-1])
            self.uniform_Tu = uniform_Tu

        self.register_buffer('lambda_tilde_1', lambda_tilde_1)
        self.register_buffer('lambda_tilde_2', lambda_tilde_2)
        
        self.marginal_m = None # need to be computed in compute_F
        self.marginal_v = None

        identity = eye_like(self.Xu, self.n_inducing_points)
        u_scale_tril = identity.repeat(self.out_dims, 1, 1)
        self.u_scale_tril = Parameter(u_scale_tril)
        
            
    def KL_prior(self, importance_weighted):
        """
        Compute -log Z + log approx likelihood
        """
        m = self.marginal_m
        v = self.marginal_v
        log_Z = self.log_Z
        
        y_tilde = self.lambda_tilde_1/self.lambda_tilde_2
        v_tilde = -.5/self.lambda_tilde_2**2
        
        approx_ll = -.5 * (torch.log(2*np.pi * v_tilde) + ( (y_tilde - m)**2 + v )/v_tilde)
        return approx_ll - log_Z
    
    
    def constrain(self):
        #torch.clamp(self.lambda_tilde_2, min=1e-12)
        return
    
    def evaluate_at(self, t):
        return
    
    
    def sample_prior(self):
        return
    
    
    def sample_posterior(self):
        return
    
        
    def compute_F(self, XZ):
        """
        Computes :math:`p(f_i(x)|y)` using Kalman filtering and RTS smoothing, giving diagonal elements
        of the posterior
        """
        XZ = self._XZ(XZ)
        
        #y_tilde = self.lambda_tilde_1/self.lambda_tilde_2
        #v_tilde = -.5/self.lambda_tilde_2**2
        
        timesteps = XZ.shape[-2]
        t = XZ[..., 0].view(-1, timesteps) # mc x trials x out_dims, time
        
        dt = self.Tu[:, 1:]-self.Tu[:-1]
        
        F, Qc, L, H = self.state_space.get_state_space()
        P0 = self.state_space.compute_covariance(F, Qc, L)
        
        # get state space dynamics
        A = self.state_space.compute_transition(dt, F)
        if A.shape[0] != t.shape[0]: # trials or mc > 1
            A = A.repeat(A.shape[0] // t.shape[0], 1, 1)
            
        
        # site parameters or approximate likelihood
        y_tilde = self.lambda_tilde_1/self.lambda_tilde_2
        v_tilde = -.5/self.lambda_tilde_2**2
            
        m_f, P_f, log_Z = Kalman_filter(A, H, P0, y_tilde, v_tilde)
        m_s, m_P = RTS_smoother(A, H, P0, m_f, P_f)
        
        # store intermediate values
        self.marginal_m = m_s.view(-1, self.out_dims, t.shape[-1])
        self.marginal_v = m_P.view(-1, self.out_dims, t.shape[-1])
        self.log_Z = log_Z
        
        #self.evaluate_at(t, )
        
        return self.marginal_m + self.mean_function(XZ), self.marginal_v
    
    
    def sample_F(self, XZ, samples=1, eps=None):
        """
        Samples from the LTI state space model
        """
        return
    
    
    def nat_grad_update(self, step_size=1.):
        """
        """
        m_g = self.marginal_m.grad
        m = self.marginal_m.data
        v_g = self.marginal_v.grad
        v = self.marginal_v.data
        self.lambda_tilde_1 = (1.-step_size)*self.lambda_tilde_1 + step_size*(m_g - 2*v_g*m)
        self.lambda_tilde_2 = (1.-step_size)*self.lambda_tilde_2 + step_size*v_g
    
    

class S2CVI(_GP):
    """
    Variational Gaussian Process model with covariates for regression and latent variables.
    """
    def __init__(self, input_dims, out_dims, state_space, Tu, learn_Tu=False, uniform_Tu=False, 
                 mean=0.0, learn_mean=False, tensor_type=torch.float, jitter=1e-6, active_dims=None):
        r"""
        Using conjugate computation variational inference and state space formulation of GPs [1].
        One can evaluate the posterior at values away from inducing points using doubly sparse structure [2].
        
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param torch.Tensor inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        
        References:
        
        [1] `FAST VARIATIONAL LEARNING IN STATE-SPACE GAUSSIAN PROCESS MODELS`, 
        Paul E. Chang, William J. Wilkinson, Mohammad Emtiyaz Khan, Arno Solin (2020)
        
        [2] `Doubly Sparse Variational Gaussian Processes`, 
        Vincent Adam, Stefanos Eleftheriadis, Nicolas Durrande, Artem Artemev, James Hensman (2020)
        """
        super().__init__(input_dims, out_dims, mean, learn_mean, tensor_type, active_dims)
        
        ### kernel ###
        self.add_module("state_space", state_space)
        
        ### State space GP setup ###
        if learn_Tu:
            self.Tu = Parameter(Tu.type(self.tensor_type))
            if uniform_Tu:
                raise ValueError('Cannot have uniformly separated time points while learning time locations')
            self.dt = None # need to compute
            self.uniform_Tu = False
            
        else:
            self.register_buffer('Tu', Tu.type(self.tensor_type))
            self.register_buffer('dt', self.Tu[:, 1:]-self.Tu[:-1])
            self.uniform_Tu = uniform_Tu

        self.register_buffer('lambda_tilde_1', lambda_tilde_1)
        self.register_buffer('lambda_tilde_2', lambda_tilde_2)
        
        self.marginal_m = None # need to be computed in compute_F
        self.marginal_v = None

        identity = eye_like(self.Xu, self.n_inducing_points)
        u_scale_tril = identity.repeat(self.out_dims, 1, 1)
        self.u_scale_tril = Parameter(u_scale_tril)
    
    
    
    
    
class TT_SVGP(_GP):
    """
    A Variational Gaussian Process model with covariates for regression and latent variables.
    Kronecker product kernel version.
    """
    def __init__(self, input_dims, out_dims, kernel_TT, inducing_loc_list, nu_list, Psi_list, S_list, S_type,
                 TT_variational=True, nu_rshape_len=1, cheb_n=100, cheb_mc=10, GR_K=1, tr_mc=10, logdet_cheb=None, 
                 mean=0.0, learn_mean=False, jitter=1e-6, tensor_type=torch.float, active_dims=None):
        r"""
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param np.ndarray inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        :param tuples kernel_tuples: a tuple listing kernels, with content 
                                     (kernel_type, topology, lengthscale, variance)
        :param tuples prior_tuples: a tuple listing prior distribution, with content 
                                    (kernel_type, topology, lengthscale, variance)
        :param tuples variational_types: a tuple listing variational distributions, with content 
                                         (kernel_type, topology, lengthscale, variance)
        :param np.ndarray/nn.Module mean: initial GP mean of shape (samples, neurons, timesteps), or if nn.Module 
                                        learnable function to compute the mean given input
        :param string inv_ink: inverse link function name
        """
        super().__init__(input_dims, out_dims, tensor_type, active_dims, True)

        ### kernel ###
        if sum([len(l) for l in kernel_TT.track_dims_list]) != self.input_dims:
            ValueError('Kernel dimensions do not match expected input dimensions')
        self.add_module("kernel", kernel_TT)
        self.D_input = len(kernel_TT.K_list) # number of input kernels
        self.GR_K = GR_K # Gauss-Rademacher K
        self.jitter = jitter
        
        ### GP mean ###
        if isinstance(mean, Number):
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean, dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean, dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, np.ndarray):
            if mean.shape != torch.Size([out_dims]): # separate mean per output dimension
                raise ValueError('Mean dimensions do not match output dimensions')
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, nn.Module):
            self.add_module("mean_function", mean)
            if learn_mean is False:
                self.mean_function.requires_grad = False
                
        else:
            raise NotImplementedError('Mean type is not supported.')
        
        ### Approximate GP setup ###
        if len(inducing_loc_list) != self.D_input:
            raise ValueError('Inducing grid dimensions do not match kernel dimensions')
        for en, u in enumerate(inducing_loc_list):
            self.register_buffer('U_'+str(en), torch.tensor(u, dtype=self.tensor_type))
        self.dim_list = np.array([u.shape[-2] for u in inducing_loc_list])
        
        self.TT_variational = TT_variational
        if self.TT_variational:
            if len(nu_list) != self.D_input or len(Psi_list) != self.D_input or \
                len(S_list) % self.D_input != 0 or len(S_type) != len(S_list):
                raise ValueError('Variational dimensions do not match kernel dimensions')

            self.nu_list = nn.ParameterList([])
            for vnu in nu_list: # list over dimensions
                self.nu_list.append(Parameter(torch.tensor(vnu[None, ...], dtype=self.tensor_type)))

            self.Psi_list = nn.ParameterList([])
            for psi in Psi_list: # list over dimensions
                self.Psi_list.append(Parameter(torch.tensor(psi[None, :, None, :], dtype=self.tensor_type)))

            self.S_list = nn.ParameterList([])
            for s in S_list: # list over dimensions*k_v
                self.S_list.append(Parameter(torch.tensor(s[None, :, None, ...], dtype=self.tensor_type)))

            # construct variational parameters, (MC, out_dims, tensor_shape)
            self.nu = linalg.TT_vector([linalg.vector(v, (-1,)) for v in self.nu_list], nu_rshape_len)
            self.Psi = linalg.TT_vector([linalg.vector(v, (-1,)) for v in self.Psi_list], 1)
            self.S = []
            for k in range(len(S_list) // self.D_input):
                offset = k*self.D_input
                ll = [
                    S_type[em+offset](m, (-1,)) for em, m in enumerate(self.S_list[offset:offset+self.D_input])
                ]
                a = linalg.TT_matrix(ll, r_shape_len=1, tensor_shape_len=len(ll[0].tensor_shape))
                self.S.append(a)
                
        else:
            if len(nu_list) != 1 or len(Psi_list) != 1:
                raise ValueError('Variational parameter lists should contain one element')
                
            # construct variational parameters
            self.nu = Parameter(torch.tensor(nu_list[0], dtype=self.tensor_type))
            self.Psi = Parameter(torch.tensor(Psi_list[0], dtype=self.tensor_type))
            self.S = nn.ParameterList([])
            for s in S_list:
                self.S.append(Parameter(torch.tensor(s, dtype=self.tensor_type)))
            
        # Chebyshev estimator
        if logdet_cheb is None:
            self.logdet_cheb = [False]*self.D_input
        else:
            self.logdet_cheb = logdet_cheb
        self.cheb_n = cheb_n
        self.cheb_mc = cheb_mc
        self.f_cheb = lambda x: np.log(1-x)
        
        self.cheb_d_others = np.ones((self.D_input,))
        if self.D_input > 1:
            for en in range(self.D_input):
                self.cheb_d_others[en] = np.prod(self.dim_list[np.arange(self.D_input) != en])
            
        self.tr_mc = tr_mc
        self.T_induc_grid = np.prod(self.dim_list)
        self.training_mode = False
                    
        
    def _U_list(self):
        U_list = []
        for d in range(self.D_input):
            U_list.append(getattr(self, 'U_{}'.format(d)))
        return U_list
            
            
    def KL_prior(self, importance_weighted):
        """
        Ignores neuron, computes over all the output dimensions, suits coupled models
        """
        # compute covariance matrix over inducing point grid
        U_list = self._U_list()
        KTT = self.kernel.Kronecker_Toeplitz(U_list)
        
        # Q(K, L xi)
        tr_LKL = 1
        for d in range(self.D_input):
            psi = self.Psi.tensor_list[d]()
            xi = 2.*torch.randint(2, (self.tr_mc,) + psi.shape[1:], dtype=psi.dtype, device=psi.device) - 1.
            xi = psi*xi
            for S in self.S:
                xi = S.tensor_list[d].mvm(xi)
                
            tr_LKL = tr_LKL * (xi * KTT.tensor_list[d].mvm(xi)).sum(-1).mean(0).sum()
        
        # Q(K, v)
        vQv = self.nu.scalar_prod(KTT.mvm(self.nu)).sum()
        
        # log det
        logdets = []
        for en, tt_ in enumerate(KTT.tensor_list):
            if self.logdet_cheb[en]:
                # surrogate loss
                xi = 2.*torch.randint(2, (self.tr_mc,) + psi.shape[1:], dtype=psi.dtype, device=psi.device) - 1.
                Kinv_xi = linalg.cg_batch(tt_().data, xi)
                K_xi = tt_.mvm(xi)
                logdet = (Kinv_xi * K_xi).sum(-1).mean(0)
                
                if self.training_mode is False: # true_loss
                    UB = tt_.full_vector().sum(-1) # Toeplitz primary vector
                    logdet_ = linalg.log_det_Chebyshev(tt_, UB, self.cheb_mc, self.f_cheb, self.cheb_n, recursive=True)
                    logdet.data = logdet_
                
            else:
                M = tt_.explicit()
                M.view(-1, M.shape[-1]**2)[:, ::M.shape[-1]+1] += self.jitter
                L = torch.linalg.cholesky(M)
                logdet = 2*torch.log(L[..., np.arange(tt_.n), np.arange(tt_.n)]).sum(-1)
                
            logdets.append(self.cheb_d_others[en]*logdet)
        logdet = torch.stack(logdets).sum(0).mean(0).sum() # sum over n_chev, mean over cheb_mc, sum over out_dims and r_shape=(1,)
        
        # log Psi trace
        tr_Psi = 1
        for t in self.Psi.tensor_list:
            tr_Psi = tr_Psi * (torch.log(t())).sum(-1)
        tr_Psi = tr_Psi.sum()
        
        KL = 0.5 * (vQv + tr_LKL - logdet - self.T_induc_grid) - tr_Psi
        return -KL
    
    
    def constrain(self):
        """
        PSD constraint on Psi
        Lower triangular constraint on S
        """
        with torch.no_grad():
            for v in self.Psi.tensor_list:
                v.tensor = torch.clamp(v(), min=1e-12)

            for s in self.S:
                for m in s.tensor_list:
                    m.constrain_tril(I=True)
            
        
    def compute_F(self, XZ):
        """
        :param X: input regressors with shape (samples, timesteps, dims)
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        XZ = self._XZ(XZ)
        U_list = self._U_list()
        
        XZ_expanded = XZ[:, None, ...].repeat(ll_samples, *((1,)*(len(XZ.shape))))
        kXU = self.kernel.X_U_list(XZ_expanded, U_list) # XZ to (K, N, N_xz)
        vnu = kXU.mvm(self.nu)
        
        cov = None
        return vnu.trace_i() + self.mean_function(XZ), cov
    
    
    def sample_F(self, XZ, ll_samples):
        """
        """
        XZ = self._XZ(XZ)
        U_list = self._U_list()
        
        # sample TT random tensors
        mc = ll_samples*XZ.shape[0]
        batch_shape = (self.out_dims,)
        random_G = linalg.sample_Gauss_Rademacher(batch_shape, mc, self.dim_list, self.GR_K, 
                                                  dtype=self.tensor_type, device=XZ.device)
        XZ_expanded = XZ[:, None, ...].repeat(ll_samples, *((1,)*len(XZ.shape)))
        kXU = self.kernel.X_U_list(XZ_expanded, U_list) # XZ to (K, N, N_xz)
        
        G = self.Psi.hadamard_prod(random_G)
        for S in self.S:
            G = S.mvm(G)
            
        G = kXU.mvm(G)
        vnu = kXU.mvm(self.nu)
        #print(G.trace_i())
        return G.trace_i() + vnu.trace_i() + self.mean_function(XZ)
    
    
    
    
class SKI_SVGP(_GP):
    """
    Sparse Variational Gaussian Process model with covariates for regression and latent variables.
    """
    def __init__(self, input_dims, out_dims, kernel_tuple, inducing_points, mean=0.0, learn_mean=False, 
                 MAP=False, whiten=False, tensor_type=torch.float, jitter=1e-6, active_dims=None):
        r"""
        Specify the kernel type with corresponding dimensions and arguments.
        Note that using shared_kernel_params=True uses Pyro's kernel module, which sets 
        hyperparameters directly without passing through a nonlinearity function.
        
        .. math::
            [f, u] &\sim \mathcal{GP}(0, k([X, X_u], [X, X_u])),\\
            y & \sim p(y) = p(y \mid f) p(f),

        where :math:`p(y \mid f)` is the likelihood.

        We will use a variational approach in this model by approximating :math:`q(f,u)`
        to the posterior :math:`p(f,u \mid y)`. Precisely, :math:`q(f) = p(f\mid u)q(u)`,
        where :math:`q(u)` is a multivariate normal distribution with two parameters
        ``u_loc`` and ``u_scale_tril``, which will be learned during a variational
        inference process.

        .. note:: The sparse model has :math:`\mathcal{O}(NM^2)` complexity for training,
            :math:`\mathcal{O}(M^3)` complexity for testing. Here, :math:`N` is the number
            of train inputs, :math:`M` is the number of inducing inputs. Size of
            variational parameters is :math:`\mathcal{O}(M^2)`.
        
        
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param np.ndarray inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        :param tuples kernel_tuples: a tuple listing kernels, with content 
                                     (kernel_type, topology, lengthscale, variance)
        :param tuples prior_tuples: a tuple listing prior distribution, with content 
                                    (kernel_type, topology, lengthscale, variance)
        :param tuples variational_types: a tuple listing variational distributions, with content 
                                         (kernel_type, topology, lengthscale, variance)
        :param np.ndarray/nn.Module mean: initial GP mean of shape (samples, neurons, timesteps), or if nn.Module 
                                        learnable function to compute the mean given input
        :param string inv_ink: inverse link function name
        """
        super().__init__(input_dims, out_dims, tensor_type, active_dims)
        self.jitter = jitter
        self.MAP = MAP
        #self.shared_kernel_params = shared_kernel_params
        
        ### kernel ###
        kernel, track_dims, constrain_dims = kernel_tuple
        if track_dims != self.input_dims:
            ValueError('Kernel dimensions do not match expected input dimensions')
        #if kernel.tensor_type != self.tensor_type:
         #   ValueError('Kernel tensor type does not match model tensor type')
        self.constrain_dims = constrain_dims
        self.add_module("kernel", kernel)
        
        ### GP mean ###
        if isinstance(mean, Number):
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean, dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean, dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, np.ndarray):
            if mean.shape != torch.Size([out_dims]):
                raise ValueError('Mean dimensions do not match output dimensions')
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, nn.Module):
            self.add_module("mean_function", mean)
            if learn_mean is False:
                self.mean_function.requires_grad = False
                
        else:
            raise NotImplementedError('Mean type is not supported.')
        
        ### Approximate GP setup ###
        if inducing_points.shape[-1] != input_dims:
            raise ValueError('Inducing point dimensions do not match expected dimensions')
        inducing_points = torch.tensor(inducing_points, dtype=self.tensor_type)
        self.n_inducing_points = inducing_points.size(-2)

        self.whiten = whiten
        self.Xu = Parameter(inducing_points)

        u_loc = self.Xu.new_zeros((self.out_dims, self.n_inducing_points))
        self.u_loc = Parameter(u_loc)

        if MAP:
            self.u_scale_tril = None
        else:
            identity = eye_like(self.Xu, self.n_inducing_points)
            u_scale_tril = identity.repeat(self.out_dims, 1, 1)
            self.u_scale_tril = Parameter(u_scale_tril)
        
            
    def KL_prior(self, importance_weighted):
        """
        Ignores neuron, computes over all the output dimensions, suits coupled models
        """
        if self.MAP: # log p(u)
            zero_loc = self.Xu.new_zeros(self.u_loc.shape)
            if self.whiten:
                identity = eye_like(self.Xu, zero_loc.shape[1])[None, ...].repeat(zero_loc.shape[0], 1, 1)
                p = distributions.Rn_MVN(zero_loc, scale_tril=identity)
            else: # loc (N, N_u), cov (N, N_u, N_u)
                p = distributions.Rn_MVN(zero_loc, scale_tril=self.Luu)
                
            return p.log_prob(self.u_loc).sum()
        
        else: # log p(u)/q(u)
            zero_loc = self.u_loc.new_zeros(self.u_loc.shape)
            if self.whiten:
                identity = eye_like(self.u_loc, zero_loc.shape[1])[None, ...].repeat(zero_loc.shape[0], 1, 1)
                p = distributions.Rn_MVN(zero_loc, scale_tril=identity)#.to_event(zero_loc.dim() - 1)
            else: # loc (N, N_u), cov (N, N_u, N_u)
                p = distributions.Rn_MVN(zero_loc, scale_tril=self.Luu)#.to_event(zero_loc.dim() - 1)

            q = distributions.Rn_MVN(self.u_loc, scale_tril=self.u_scale_tril)#.to_event(self.u_loc.dim()-1)
            kl = torch.distributions.kl.kl_divergence(q, p).sum() # sum over neurons
            if torch.isnan(kl).any():
                kl = 0.
                print('Warning: sparse GP prior is NaN, ignoring prior term.')
            return -kl
    
    
    def constrain(self):
        # constrain topological inducing points of sphere
        for k, n in self.constrain_dims:
            L2 = self.Xu[..., k:k+n].data.norm(2, -1)[..., None]
            self.Xu[..., k:k+n].data /= L2
        
        if self.u_scale_tril is not None: # constrain K to be PSD, L diagonals > 0 as needed for det K
            self.u_scale_tril.data = torch.tril(self.u_scale_tril.data)
            Nu = self.u_scale_tril.shape[-1]
            self.u_scale_tril.data[:, np.arange(Nu), np.arange(Nu)] = \
                torch.clamp(self.u_scale_tril.data[:, np.arange(Nu), np.arange(Nu)], min=1e-12)
            
        
    def compute_F(self, XZ):
        """
        Computes :math:`p(f(x)|u)` and also updating :math:`L_{uu}` matrix
        model call uses :math:`L_{uu}` for the MVN, call after this function
        
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        
        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).
            
        .. note:: Variational parameters ``u_loc``, ``u_scale_tril``, the
            inducing-point parameter ``Xu``, together with kernel's parameters have
            been learned.
        
        covariance_type is a flag to decide if we want to predict full covariance matrix or 
        just variance.
        
        .. note:: The GP is centered around zero with fixed zero mean, but a learnable 
            mean is added after computing the posterior to get the mapping mean.
        
        XZ # K, N, T, D
        X_u = self.Xu[None, ...] # K, N, T, D
        
        :param X: input regressors with shape (samples, timesteps, dims)
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor), both of shape ()
        """
        XZ = self._XZ(XZ) 
        loc, cov, self.Luu = p_F_U(
            XZ, self.Xu, self.kernel, self.u_loc, self.u_scale_tril, compute_cov=~self.MAP, 
            full_cov=False, whiten=self.whiten, jitter=self.jitter
        )
        
        return loc + self.mean_function(XZ), cov
    
    
    def sample_F(self, XZ, eps=None):
        """
        Samples from the variational posterior.
        """
        XZ = self._XZ(XZ)
        loc, cov, self.Luu = p_F_U(XZ, XZ, self.kernel, self.u_loc, self.u_scale_tril,
                                   full_cov=True, whiten=self.whiten, jitter=self.jitter)
        
        if eps is None: # sample random vector
            eps = torch.randn(XZ.shape, dtype=self.tensor_type, device=cov.device)
            
        return loc + self.mean_function(XZ) + \
               (torch.linalg.cholesky(cov) * eps[..., None, :]).sum(-1)
    
    
    
    
    
class ST_SVGP(_GP):
    """
    Spatio-temporal Gaussian processes
    """
    def __init__(self, input_dims, out_dims, temporal_kernel, spatial_kernel, Tu, spat_inducing_points, 
                 mean=0.0, learn_mean=False, tensor_type=torch.float, active_dims=None):
        r"""
        :param int out_dims: number of output dimensions of the GP, usually neurons
        :param np.ndarray inducing_points: initial inducing points with shape (neurons, n_induc, input_dims)
        :param tuples kernel_tuples: a tuple listing kernels, with content 
                                     (kernel_type, topology, lengthscale, variance)
        :param tuples prior_tuples: a tuple listing prior distribution, with content 
                                    (kernel_type, topology, lengthscale, variance)
        :param tuples variational_types: a tuple listing variational distributions, with content 
                                         (kernel_type, topology, lengthscale, variance)
        :param np.ndarray/nn.Module mean: initial GP mean of shape (samples, neurons, timesteps), or if nn.Module 
                                        learnable function to compute the mean given input
        :param string inv_ink: inverse link function name
        
        [1] `Spatio-Temporal Variational Gaussian Processes`, 
        Oliver Hamelijnck, William J. Wilkinson, Niki A. Loppi, Arno Solin, Theodoros Damoulas (2021)
        """
        super().__init__(input_dims, out_dims, mean, learn_mean, tensor_type, active_dims, True)

        ### kernel ###
        kernel, track_dims = kernels.TT_kernel(kernel_tuples, kern_f, self.tensor_type)
        if track_dims != self.input_dims:
            ValueError('Kernel dimensions do not match expected input dimensions')
        self.add_module("kernel", kernel)
        
        ### GP mean ###
        if isinstance(mean, Number):
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean, dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean, dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, np.ndarray):
            if mean.shape != torch.Size([out_dims]): # separate mean per output dimension
                raise ValueError('Mean dimensions do not match output dimensions')
            if learn_mean:
                self.mean = Parameter(torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            else:
                self.register_buffer('mean', torch.tensor(mean[None, :, None], dtype=self.tensor_type))
            self.mean_function = lambda x: self.mean
            
        elif isinstance(mean, nn.Module):
            self.add_module("mean_function", mean)
            if learn_mean is False:
                self.mean_function.requires_grad = False
                
        else:
            raise NotImplementedError('Mean type is not supported.')
        
        ### Approximate GP setup ###
        self.n_inducing_points = inducing_points.size(-2)
            
        self.Xu = Parameter(inducing_points)
        
        self.Toeplitz_covariance = Toeplitz_covariance
        self.Psi = 1
        if Toeplitz_covariance:
            self.S = 1
            self.L = 1
            
            L_ = []
            for _ in range(2):
                a = torch.zeros(2*d-1)
                a[-d+1:] = torch.randn(d-1).float()
                a[-d] = 1
                L = explicit_Toeplitz(a, d)
                L_.append(L)

            psi = torch.zeros((d, d))
            psi[torch.arange(d), torch.arange(d)] = torch.randn(d)

            a = torch.zeros(2*d-1)
            a[-d+1:] = torch.randn(d-1).float()
            S = explicit_Toeplitz(a, d)
            
        else:
            identity = eye_like(self.Xu, self.n_inducing_points)
            u_scale_tril = identity.repeat(self.out_dims, 1, 1)
            self.u_scale_tril = Parameter(u_scale_tril)
            self.S = 1

        
    def KL_prior(self, importance_weighted):
        """
        Ignores neuron, computes over all the output dimensions, suits coupled models
        """
        f = lambda x: np.log(1-x)
        for K in self.kernel.tensor_list:
            logdet = linalg.log_det_Chebyshev(K, UB, 100, f, 1000, recursive=True)
            
        # Q(K, L xi)
        Lxi = self._Lxi(xi)
        
        # Q(K, v)
        
        
        KL = 1
        return KL
    
    
    def constrain(self):
        
        if self.u_scale_tril is not None: # constrain K to be PSD, L diagonals > 0 as needed for det K
            self.u_scale_tril.data = torch.tril(self.u_scale_tril.data)
            Nu = self.u_scale_tril.shape[-1]
            self.u_scale_tril.data[:, np.arange(Nu), np.arange(Nu)] = \
                torch.clamp(self.u_scale_tril.data[:, np.arange(Nu), np.arange(Nu)], min=1e-12)
            
        
    def compute_F(self, XZ):
        """
        Computes :math:`p(f(x)|u)` and also updating :math:`L_{uu}` matrix
        model call uses :math:`L_{uu}` for the MVN, call after this function
        
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        
        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, u_{loc}, u_{scale\_tril})
            = \mathcal{N}(loc, cov).
            
        .. note:: Variational parameters ``u_loc``, ``u_scale_tril``, the
            inducing-point parameter ``Xu``, together with kernel's parameters have
            been learned.
        
        covariance_type is a flag to decide if we want to predict full covariance matrix or 
        just variance.
        
        .. note:: The GP is centered around zero with fixed zero mean, but a learnable 
            mean is added after computing the posterior to get the mapping mean.
        
        
        :param X: input regressors with shape (samples, timesteps, dims)
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        #if self.shared_kernel_params:
        #    loc, cov, self.Luu = linalg.pFU_shared(self.out_dims, X, self.Xu, self.kernel, self.u_loc, self.u_scale_tril,
        #                       cov_type=self.covariance_type, whiten=self.whiten, jitter=self.jitter)
        #else:
        XZ = self._XZ(XZ)
        
        if TT_sample: # sample TT random tensors
            
            random_G = torch.randn(d)
            
        else: # get explicit from TT representation

            loc, cov, self.Lff = p_F_U(self.out_dims, XZ, self.Xf, self.kernel, self.f_loc, self.f_scale_tril, 
                                                 cov_type=self.covariance_type, whiten=False, jitter=self.jitter)
        
        return loc + self.mean_function(XZ), cov