import torch
import torch.nn as nn
import torch.fft as fft

import time
import copy

import scipy

from . import chebyshev




# operations
def _explicit_Toeplitz(a, d):
    """
    input a represents the below principal vector 
    f_{-n} ... f_0 ... f_n
    """
    if d > (a.shape[0]-1)//2+1:
        a = torch.cat((a, a), dim=-1) # ensure that for circulant we cover
    M = torch.empty((*a.shape[:-1], d, d), device=a.device, dtype=a.dtype)
    for k in range(d):
        if k == 0:
            s = None
        else:
            s = -k
        M[..., :, k] = a[..., -d-k:s]
    return M


def _circulant_from_Toeplitz(a, d):
    """
    d = (a.shape[0] - 1) // 2
    """
    return torch.cat((a[..., -d:], a[..., :a.shape[-1]-d]), dim=-1)


def _embed_sym_Toeplitz(a):
    """
    Only specify 
    """
    a = torch.cat((torch.flip(a[..., 1:-1], dims=(-1,)), a), dim=-1)
    return a
    
    
def _circulant_mvm(a, v, p=1):
    """
    To perform matrix-matrix products, transpose RHS matrix to get extra batch dimension, 
    and add singleton dimension to the LHS matrix.
    """
    v_ = torch.zeros((*v.shape[:-1], a.shape[-1]), 
                     dtype=a.dtype, device=a.device)
    v_[..., :v.shape[-1]] = v
    return fft.irfft(fft.rfft(a)**p * fft.rfft(v_), a.shape[-1]) # need to pass original length, otherwise not unique






# objects
class _tensor():
    
    def __init__(self, tensor, sample_shape, batch_shape):
        self.assign(tensor, sample_shape, batch_shape)
    
    def assign(self, tensor, sample_shape, batch_shape):
        self.tensor = tensor
        self.sample_shape = sample_shape
        self.batch_shape = batch_shape
        self.tensor_shape = tensor.shape[len(sample_shape+batch_shape):]
    
    def __call__(self):
        return self.tensor
    
    def explicit():
        raise NotImplementedError()


        
class vector(_tensor):
    """
    Used as object for input to tensor train
    """
    def __init__(self, vector, sample_shape):
        """
        (batch_shape, dim)
        """
        self.assign(vector, sample_shape)
        
    def assign(self, vector, sample_shape):
        s = len(sample_shape)
        super().assign(vector, sample_shape, vector.shape[s:-1])
    
    def explicit(self):
        return self()
    


class _matrix(_tensor):
    
    def assign(self, matrix, sample_shape, batch_shape):
        """
        The matrix may refer to the actual matrix, or column, or low rank representation X
        """
        super().assign(matrix, sample_shape, batch_shape)

    def get_eye(self):
        """
        Get identity matrix with same device and dtype as matrix
        """
        raise NotImplementedError
    
    def sum_rows(self):
        raise NotImplementedError
    
    def sum_columns(self):
        raise NotImplementedError
        

    
# matrices
class general_matrix(_matrix):
    """
    Vanilla matrix
    """
    def __init__(self, matrix, sample_shape):
        self.assign(matrix, sample_shape)
        
    def __copy__(self, matrix=None, sample_shape=None):
        if matrix is None:
            matrix = self()
        if sample_shape is None:
            sample_shape = self.sample_shape
        return type(self)(matrix, sample_shape)
    
    def assign(self, matrix, sample_shape):
        s = len(sample_shape)
        super().assign(matrix, sample_shape, matrix.shape[s:-2])
        self.n = self.tensor_shape[-2] # left matrix dimension
    
    def constrain_tril(self, I=False):
        self.tensor = torch.tril(self.tensor)
        if I:
            self.tensor.view(-1, self.n**2)[:, ::self.n+1] = 1
        
    def explicit(self):
        return self.tensor

    def mvm(self, v):
        return (self.tensor @ v[..., None])[..., 0]
    
    def get_eye(self):
        I = self.tensor.new_zeros(self.tensor.shape)
        I[..., list(range(self.n)), list(range(self.n))] = 1
        return I
    
    def sum_rows(self):
        return self.tensor.sum(-1)
    
    def sum_columns(self):
        return self.tensor.sum(-2)
    
    
class low_rank_matrix(_matrix):
    """
    Specify square matrix as X, M = X @ X.T
    Note the tr(M) = ||X||_F^2
    """
    def __init__(self, matrix, sample_shape):
        self.assign(matrix, sample_shape)
        
    def __copy__(self, matrix=None, sample_shape=None):
        if matrix is None:
            matrix = self()
        if sample_shape is None:
            sample_shape = self.sample_shape
        return type(self)(matrix, sample_shape)
    
    def assign(self, matrix, sample_shape):
        s = len(sample_shape)
        super().assign(matrix, s, matrix.shape[s:-2])
        self.n = matrix.shape[-2]
        
    def explicit(self):
        return self.tensor @ self.tensor.transpose(-2, -1)

    def mvm(self, v):
        return (self.tensor @ (self.tensor.transpose(-2, -1) @ v[..., None]))[..., 0]
    
    def get_eye(self):
        I = self.tensor.new_zeros((self.n, self.n))
        I[..., list(range(self.n)), list(range(self.n))] = 1
        return I
    
    def sum_rows(self):
        return self.explicit().sum(-1)
    
    def sum_columns(self):
        return self.explicit().sum(-2)



class Toeplitz_matrix(_matrix):
    
    def __init__(self, matrix, sample_shape, d):
        self.assign(matrix, sample_shape, d)
    
    def __copy__(self, matrix=None, sample_shape=None):
        if matrix is None:
            matrix = self()
        if sample_shape is None:
            sample_shape = self.sample_shape
        return type(self)(matrix, sample_shape, self.n)
    
    def assign(self, matrix, sample_shape, d):
        s = len(sample_shape)
        super().assign(matrix, sample_shape, matrix.shape[s:-1])
        self.n = d
    
    def constrain_tril(self, I=False):
        self.tensor[..., self.n:] = 0
        if I:
            self.tensor[..., 0] = 1
    
    def explicit(self):
        M = _explicit_Toeplitz(self.tensor, self.n)
        return M

    def mvm(self, v):
        a_c = _circulant_from_Toeplitz(self.tensor, self.n)
        return _circulant_mvm(a_c, v)[..., :self.n]
    
    def get_eye(self):
        I = self.tensor.new_zeros(self.tensor.shape)
        I[..., 0] = 1
        return I
    
    def sum_rows(self):
        m = torch.cat((self.tensor, self.tensor), dim=-1)
        return self.explicit().sum(-1)
    
    def sum_columns(self):
        m = torch.cat((self.tensor, self.tensor), dim=-1)
        return self.tensor
    
    
    
class trilI_Toeplitz_matrix(_matrix):
    
    def __init__(self, matrix, sample_shape):
        self.assign(matrix, sample_shape)
    
    def __copy__(self, matrix=None, sample_shape=None):
        if matrix is None:
            matrix = self()
        if sample_shape is None:
            sample_shape = self.sample_shape
        return type(self)(matrix, sample_shape)
    
    def assign(self, matrix, sample_shape):
        s = len(sample_shape)
        super().assign(matrix, sample_shape, matrix.shape[s:-1])
        self.n = matrix.shape[-1]+1
    
    def _expand_matrix(self):
        """
        Expand the tensor dimension to form Toeplitz vector
        """
        bs = self.tensor.shape[:-1]
        z = self.tensor.new_zeros(bs+(self.n,))
        z[..., -1] = 1 # diagonal
        return torch.cat((z, self.tensor), dim=-1)
    
    def constrain_tril(self, I=True):
        return # nothing to do here
    
    def explicit(self):
        M = _explicit_Toeplitz(self._expand_matrix(), self.n)
        return M

    def mvm(self, v):
        a_c = _circulant_from_Toeplitz(self._expand_matrix(), self.n)
        return _circulant_mvm(a_c, v)[..., :self.n]
    
    def get_eye(self):
        I = self.tensor.new_zeros(self.tensor.shape)
        return I
    
    def sum_rows(self):
        m = torch.cat((self.tensor, self.tensor), dim=-1)
        return self.explicit().sum(-1)
    
    def sum_columns(self):
        m = torch.cat((self.tensor, self.tensor), dim=-1)
        return self.tensor

    

class sym_Toeplitz_matrix(Toeplitz_matrix):
    
    def __init__(self, matrix, sample_shape):
        self.assign(matrix, sample_shape)
        
    def __copy__(self, matrix=None, sample_shape=None):
        if matrix is None:
            matrix = self()
        if sample_shape is None:
            sample_shape = self.sample_shape
        return type(self)(matrix, sample_shape)
    
    def assign(self, matrix, sample_shape):
        s = len(sample_shape)
        super().assign(matrix, sample_shape, matrix.shape[s:-1])
        self.n = matrix.shape[-1]
        
    def full_vector(self):
        return _embed_sym_Toeplitz(self.tensor)
        
    def explicit(self):
        a_s = self.full_vector()
        M = _explicit_Toeplitz(a_s, self.n)
        return M
    
    def mvm(self, v):
        #assert self.n == v.shape[-1]
        a_s = self.full_vector()
        a_c = _circulant_from_Toeplitz(a_s, self.n)
        return _circulant_mvm(a_c, v)[..., :self.n]
    
    def sum_rows(self):
        a_s = self.full_vector()
        # parallelized or not?
        return self.tensor.sum(-1)
    
    def sum_columns(self):
        return self.sum_rows()
    
    
    
class circulant_matrix(Toeplitz_matrix):
    
    def __init__(self, matrix, sample_shape):
        self.assign(matrix, sample_shape)
        
    def __copy__(self, matrix=None, sample_shape=None):
        if matrix is None:
            matrix = self()
        if sample_shape is None:
            sample_shape = self.sample_shape
        return type(self)(matrix, sample_shape)
    
    def assign(self, matrix, sample_shape):
        s = len(sample_shape)
        super().assign(matrix, sample_shape, matrix.shape[s:-1])
        self.n = matrix.shape[-1]
        
    def explicit(self):
        M = _explicit_Toeplitz(self.tensor, self.n)
        return M
    
    def mvm(self, v, p=1):        
        return _circulant_mvm(self.tensor, v, p)
    
    def sum_rows(self):
        return self.tensor.sum(-1)
    
    def sum_columns(self):
        return self.sum_rows()
    
    
    
    
# TT format
class TT_list():
    """
    Tensor elements are not necessarily of same shape or r_shape here (most general TT-format).
    """
    def __init__(self, tensor_list, r_shape_len, tensor_shape_len):
        """
        tensor_list consist of the tensors with tensor core shapes (batch_shape, r_shape, tensor_shape)
        These can be _matrix objects, r_shape is treated as batch dimensions within it
        Note batch_shape in tensor objects imply the batch_shape and r_shape of the TT format.
        """
        self.tensor_list = tensor_list
        self.D = len(tensor_list)
        self.tensor_shape_len = tensor_shape_len
        self.r_shape_len = r_shape_len # has to be shared for all tensors
        
        t_0 = tensor_list[0]
        self.sample_shape = t_0.sample_shape
        self.batch_shape = t_0.batch_shape[:-r_shape_len]
        for t in tensor_list[1:]:
            if self.batch_shape != t.batch_shape[:-r_shape_len]:
                raise ValueError('Batch shapes do not match of individual TT cores')
            if self.sample_shape != t.sample_shape:
                raise ValueError('Sample shapes do not match of individual TT cores')
                
        self.bs = self.sample_shape + self.batch_shape
        self.dtype = t_0().dtype
        self.device = t_0().device
    
    def add(self, TT):
        """
        Addition
        """
        """
        Supports diagonal cores
        shape of (batch_shape, r, d, n)
        """
        raise NotImplementedError
        
    def hadamard_prod(self, TT):
        raise NotImplementedError 
        
    def scalar_prod(self, TT):
        """
        V is also in TT-format
        """
        raise NotImplementedError 
        
    def trace_i(self):
        """
        Return the TT trace.
        """
        raise NotImplementedError 
    
    def explicit(self):
        """
        Returns the fully expanded tensor
        """
        raise NotImplementedError 
        
    def get_info(self):
        """
        Get the TT-format decomposition from PyTorch tensor
        """
        tensor_shapes = [T.shape[-tensor_shape_len:] for T in self.tensor_list]
        r_shapes = [T.shape[-tensor_shape_len-1-r_shape_len:-tensor_shape_len-1] for T in self.tensor_list]
        return r_shapes, tensor_shapes

    
    
class TT_vector(TT_list):
    """
    """
    def __init__(self, tensor_list, r_shape_len):
        super().__init__(tensor_list, r_shape_len, 1)
        
    def add(self, TT):
        if self.r_shape_len == 1: # diagonal cores
            t_list = []
            for en, t in enumerate(self.tensor_list):
                t_list.append(vector(torch.cat((t, TT.tensor_list[en]), dim=-tensor_shape_len-1), self.sample_shape))
            return TT_vector(t_list, 1)
        
        elif self.r_shape_len == 2:
            t_list = []
            for en, t_ in enumerate(self.tensor_list):
                t = t_()
                tt = TT.tensor_list[en]()
                
                ts = t.shape[-self.tensor_shape_len:]
                rs = t.shape[-self.tensor_shape_len-2:-self.tensor_shape_len]
                rs_ = tt.shape[-self.tensor_shape_len-2:-self.tensor_shape_len]
                
                rp_shape = (rs.shape[0]+rs_.shape[0], rs.shape[1]+rs_.shape[1])
                T = torch.zeros(self.bs + rp_shape + ts, device=self.device, dtype=self.dtype)
                T[..., :rs.shape[0], :rs.shape[1], ts] = t
                T[..., rs.shape[0]:, rs.shape[1]:, ts] = tt
                t_list.append(vector(T, self.sample_shape))
            return TT_vector(t_list, 2)
        
    def hadamard_prod(self, TT):
        """
        Returns the Hadamard product
        """
        if self.r_shape_len == 1: # diagonal cores
            T_list = []
            for en, t_ in enumerate(self.tensor_list):
                t = t_()
                vl = t.shape[-1]
                tt = TT.tensor_list[en]()
                v = tt.view(self.bs + (tt.shape[-2], 1, vl))*t.view(self.bs + (1, t.shape[-2], vl))
                T_list.append(vector(v.view(self.bs + (tt.shape[-2]*t.shape[-2], vl)), self.sample_shape))
            return TT_vector(T_list, 1)
                       
        elif self.r_shape_len == 2: # full cores
                       
            return (V[..., None, :, :]*W[..., None, :, :, :]).view(*V.shape[:-3], -1, *V.shape[-2:])
        
    def scalar_prod(self, TT):
        """
        V is also in TT-format
        """
        TT_ = self.hadamard_prod(TT)
        
        if self.r_shape_len == 1: # diagonal cores
            T_list = []
            v = TT_.tensor_list[0]().sum(-1)
            for en, t_ in enumerate(TT_.tensor_list[1:]):
                v *= t_().sum(-1)
            return v.sum(-1)
    
    def trace_i(self):
        """
        Return the TT trace of T = prod G[ij]
        """
        L = self.tensor_list[0]().shape[-1]
        v = self.tensor_list[0]()
        
        if self.r_shape_len == 1: # diagonal cores
            for t in self.tensor_list[1:]:
                v *= t()[..., :L]
            return  v.sum(-2)
        
        elif self.r_shape_len == 2: # full cores
            for t in self.tensor_list[1:]:
                v *= t()[..., :L]
            return  v.sum(-2)
    
    def explicit(self):
        """
        Returns the fully expanded tensor
        """
        if self.r_shape_len == 1: # diagonal cores
            t = self.tensor_list[0]()
            rs = t.shape[-self.tensor_shape_len-1]
            dp = t.shape[-1]
            v = t.view(self.bs + (rs, t.shape[-1],) + (1,)*(self.D-1))
            for en, t_ in enumerate(self.tensor_list[1:]):
                tt = t_()
                dp *= tt.shape[-1]
                v = v*tt.view(self.bs + (rs,) + (1,)*(en+1) + (tt.shape[-1],) + (1,)*(self.D-en-2))
            
            return v.view(*self.bs, rs, dp).sum(-2)
        
        elif self.r_shape_len == 2: # full cores
            rs = self.r_shape_len
            bsl = len(self.bs)+self.r_shape_len
            t = self.tensor_list[0]()
            dp = t.shape[-1]
            v = t.view(self.bs + (t.shape[-1],) + (1,)*(self.D-1))
            for en, t in enumerate(self.tensor_list[1:]):
                v = (v.view(v.shape[:bsl] + (1,) + v.shape[bsl:])*t.view(bs + (1,) + rs + (1,)*(en+1) + (-1,) + (1,)*(self.D-en-2))).sum(bsl-1)
            
            return v.view(*self.batch_shape, -1)
    
        
        
class TT_matrix(TT_list):
    """
    """        
    def mvm(self, TTv):
        """
        TTv is vector in tensor train list class
        """
        if self.r_shape_len == 1: # diagonal cores
            t_list = []
            for en, T in enumerate(self.tensor_list):
                ttv = TTv.tensor_list[en]
                r1 = T.batch_shape[-1]
                r2 = ttv.batch_shape[-1]
                
                T.assign(T().unsqueeze(-self.tensor_shape_len-1), T.sample_shape)
                t_list.append(vector(
                    T.mvm(ttv().unsqueeze(-TTv.tensor_shape_len-2)).view(*self.bs, r1*r2, T.n), 
                    T.sample_shape
                ))
                T.assign(T().squeeze(-self.tensor_shape_len-1), T.sample_shape)
                
            return TT_vector(t_list, 1)
        
    def explicit(self):
        """
        Returns the fully expanded tensor
        """
        if self.r_shape_len == 1: # diagonal cores
            t = self.tensor_list[0].explicit()
            rs = t.shape[-3] # matrix always has 2 tensor dimensions
            tot_d = t.shape[-1]
            
            d_ = (t.shape[-1],) + (1,)*(self.D-1)
            v = t.view(self.bs + (rs,) + d_ + d_)
            for en, t_ in enumerate(self.tensor_list[1:]):
                tt = t_.explicit()
                tot_d *= tt.shape[-1]
                
                d_ = (1,)*(en+1) + (tt.shape[-1],) + (1,)*(self.D-en-2)
                v = v*tt.view(self.bs + (rs,) + d_ + d_)
            
            return v.view(*self.bs, rs, tot_d, tot_d).sum(-3)
        
        elif self.r_shape_len == 2: # full cores
            rs = self.r_shape_len
            bsl = len(self.bs)+self.r_shape_len
            
            t = self.tensor_list[0]
            v = t.view(self.bs + (-1,) + (1,)*(self.D-1))
            for en, t in enumerate(self.tensor_list[1:]):
                v = (v.view(v.shape[:bsl] + (1,) + v.shape[bsl:])*t.view(self.bs + (1,) + rs + (1,)*(en+1) + (-1,) + (1,)*(self.D-en-2))).sum(bsl-1)
            
            return v.view(*self.batch_shape, -1)

        
        
def sample_Gauss_Rademacher(batch_shape, mc, dim_list, K, dtype=torch.float, device='cpu'):
    """
    """
    bs = (mc,) + batch_shape
    D = len(dim_list)
    T = []
    for d in range(D):
        dim = dim_list[d]
        T_ = torch.empty(bs + (K*D, dim), dtype=dtype, device=device)
        T_[..., d*K:(d+1)*K, :] = torch.randn(bs + (K, dim), dtype=dtype, device=device)/torch.sqrt(torch.tensor(float(D*K)))
        if d > 0:
            T_[..., :d*K, :] = 2*torch.randint(2, bs + (K*d, dim), dtype=dtype, device=device)-1
        if d < D-1:
            T_[..., (d+1)*K:, :] = 2*torch.randint(2, bs + (K*(D-1-d), dim), dtype=dtype, device=device)-1
        T.append(vector(T_, (-1,)))
        
    return TT_vector(T, 1)
                       
                       

def r_one_to_r_two(TT):
    """
    Convert r_shape_len from 1 to 2 (full core representation)
    """
    for t in TT.tensor_list:
        t()
    TT_ = type(TT)
    return TT_
        



# stochastic estimators
def Hutchinson_trace(K, mc):
    """
    Trace estimator
    """
    d = K().shape[-1]
    batch_shape = K().shape[:-1]
    K.assign(K().unsqueeze(0))
    G = torch.randint(2, size=(mc, *batch_shape, d), dtype=K().dtype, device=K().device)
    G = G*2-1
    tr = (G*K.mvm(G)).sum(-1).mean(0)
    K.assign(K().squeeze(0))
    return tr



def Chebyshev_estimator_recursive(A, x, f, n):
    """
    Compute the intermediate terms recursively
    """
    c = chebyshev.Chebyshev_coeff(f, n)
    c = torch.tensor(c, dtype=A().dtype, device=A().device)
    c = c.view(n+1, *((1,)*(len(x.shape[:-1]))))
    
    A.assign(A().unsqueeze(0), (-1,))
    An_x = torch.empty((n+1, *x.shape), dtype=A().dtype, device=A().device)

    An_x[0, ...] = x#.new_ones(x.shape)
    An_x[1, ...] = A.mvm(x)
    for k in range(2, n+1):
        a = 2 * (A.mvm(An_x[k-1, ...]))
        An_x[k, ...] = a - An_x[k-2, ...]
    A.assign(A().squeeze(0), (-1,))

    return (c * (x[None, ...] * An_x).sum(-1)).sum(0).mean(0)
    #return (c.view(n, *torch.ones(len(x.shape[:-3]))) * TT_scalar_product(x[None, ...], An_x))



def Chebyshev_estimator(A, x, f, n):
    """
    Compute in the basis of matrix powers, not recursive formula
    """
    b = chebyshev.matrix_power_basis_coeff(n, chebyshev.Chebyshev_coeff(f, n))
    b = torch.tensor(b, dtype=A().dtype, device=A().device)
    b = b.view(n+1, *((1,)*(len(x.shape[:-1]))))

    A.assign(A().unsqueeze(0), (-1,))
    An_x = torch.empty((n+1, *x.shape), dtype=A().dtype, device=A().device)
    An_x[0, ...] = x
    for k in range(1, n+1):
        An_x[k, ...] = A.mvm(An_x[k-1, ...])
    A.assign(A().squeeze(0), (-1,))
    
    return (b * (x[None, ...] * An_x).sum(-1)).sum(0).mean(0)
    #return (b * TT_scalar_product(x[None, ...], An_x)).sum(0)



def log_det_Chebyshev(K, UB, mc, f, n, recursive=False):
    """
    A is 1 - K~, K~ has eigenvalues between 0 and 1
    K is assumed to be Kronecker, so no r_shape
    """
    A = K.__copy__(K.get_eye() - K()/UB[..., None])
    
    G = torch.randint(2, size=(mc, *K.batch_shape, K.n), dtype=K().dtype, device=K().device)
    G = G*2-1
    
    if recursive:
        logdet = Chebyshev_estimator_recursive(A, G, f, n)
    else:
        logdet = Chebyshev_estimator(A, G, f, n)

    return logdet + torch.log(UB)*K.n




# control    
class solve_continuous_lyapunov(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx, ):
        return scipy.linalg.solve_continuous_lyapunov()

    @staticmethod
    def backward(ctx, output_grad):
        return
    
    
    
class solve_discrete_lyapunov(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx, ):
        scipy.linalg.solve_continuous_lyapunov()
        return

    @staticmethod
    def backward(ctx, output_grad):
        return
    
    
    
class solve_continuous_riccati(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx, ):
        scipy.linalg.solve_continuous_lyapunov()
        return

    @staticmethod
    def backward(ctx, output_grad):
        return
    
    
    
class solve_discrete_riccati(torch.autograd.Function):
    """
    """
    @staticmethod
    def forward(ctx, ):
        scipy.linalg.solve_continuous_lyapunov()
        return

    @staticmethod
    def backward(ctx, output_grad):
        return
    
    
    
# matrix
class matrix_exponential(torch.autograd.Function):
    """
    Compute P^-1 B, where P is a PSD matrix, using the Cholesky factoristion
    """
    @staticmethod
    def forward(ctx, X):
        with torch.enable_grad():
            ctx = X
            return torch.matrix_exponential(X)
        
    @staticmethod
    def backward(ctx, output_grad):
        return X.grad
    

    
# solves
def PSD_solve(P, B):
    """
    Compute P^-1 B, where P is a PSD matrix, using the Cholesky factoristion
    """
    L = torch.cholesky(P)
    return torch.triangular_solve(Q, L, upper=False)[0]
    
    


# CG
def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False,
             resetiter=1, lr=1):
    """
    Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.
    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, m, n = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert X0.shape == (K, m, n)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0 # initial x
    
    B_norm = torch.norm(B, dim=-1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    reset = True
    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        
        if reset:
            R_k = B - A_bmm(X_k) # residual
            Z_k = M_bmm(R_k) # pre-conditioned residual

            P_k = torch.zeros_like(Z_k)

            P_k1 = P_k
            R_k1 = R_k
            R_k2 = R_k
            X_k1 = X0
            Z_k1 = Z_k
            Z_k2 = Z_k
        
            Z_k = M_bmm(R_k)
            
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
            
            reset = False
            kk = 1
            
        else:
            Z_k = M_bmm(R_k)
            
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(-1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(-1) / denominator
            P_k = Z_k1 + beta.unsqueeze(-1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(-1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(-1) / denominator * lr
        X_k = X_k1 + alpha.unsqueeze(-1) * P_k
        R_k = R_k1 - alpha.unsqueeze(-1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=-1)

        if kk >= resetiter:
            reset = True
        else:
            kk += 1
        
        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


class CG(torch.autograd.Function):

    def __init__(self, A_bmm, M_bmm=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        self.A_bmm = A_bmm
        self.M_bmm = M_bmm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose

    def forward(self, B, X0=None):
        X, _ = cg_batch(self.A_bmm, B, M_bmm=self.M_bmm, X0=X0, rtol=self.rtol,
                     atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return X

    def backward(self, dX):
        dB, _ = cg_batch(self.A_bmm, dX, M_bmm=self.M_bmm, rtol=self.rtol,
                      atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return dB
    
    
    
"""GPyTorch

import warnings

import torch

from .. import settings
from .deprecation import bool_compat
from .warnings import NumericalWarning
"""

def _default_preconditioner(x):
    return x.clone()


@torch.jit.script
def _jit_linear_cg_updates(
    result, alpha, residual_inner_prod, eps, beta, residual, precond_residual, mul_storage, is_zero, curr_conjugate_vec
):
    # # Update result
    # # result_{k} = result_{k-1} + alpha_{k} p_vec_{k-1}
    result = torch.addcmul(result, alpha, curr_conjugate_vec, out=result)

    # beta_{k} = (precon_residual{k}^T r_vec_{k}) / (precon_residual{k-1}^T r_vec_{k-1})
    beta.resize_as_(residual_inner_prod).copy_(residual_inner_prod)
    torch.mul(residual, precond_residual, out=mul_storage)
    torch.sum(mul_storage, -2, keepdim=True, out=residual_inner_prod)

    # Do a safe division here
    torch.lt(beta, eps, out=is_zero)
    beta.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, beta, out=beta)
    beta.masked_fill_(is_zero, 0)

    # Update curr_conjugate_vec
    # curr_conjugate_vec_{k} = precon_residual{k} + beta_{k} curr_conjugate_vec_{k-1}
    curr_conjugate_vec.mul_(beta).add_(precond_residual)


@torch.jit.script
def _jit_linear_cg_updates_no_precond(
    mvms,
    result,
    has_converged,
    alpha,
    residual_inner_prod,
    eps,
    beta,
    residual,
    precond_residual,
    mul_storage,
    is_zero,
    curr_conjugate_vec,
):
    torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
    torch.sum(mul_storage, dim=-2, keepdim=True, out=alpha)

    # Do a safe division here
    torch.lt(alpha, eps, out=is_zero)
    alpha.masked_fill_(is_zero, 1)
    torch.div(residual_inner_prod, alpha, out=alpha)
    alpha.masked_fill_(is_zero, 0)

    # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
    alpha.masked_fill_(has_converged, 0)

    # Update residual
    # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
    torch.addcmul(residual, -alpha, mvms, out=residual)

    # Update precond_residual
    # precon_residual{k} = M^-1 residual_{k}
    precond_residual = residual.clone()

    _jit_linear_cg_updates(
        result,
        alpha,
        residual_inner_prod,
        eps,
        beta,
        residual,
        precond_residual,
        mul_storage,
        is_zero,
        curr_conjugate_vec,
    )


def linear_cg(
    matmul_closure,
    rhs,
    n_tridiag=0,
    tolerance=1e-12,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=None,
    max_tridiag_iter=None,
    initial_guess=None,
    preconditioner=None, 
    terminate_cg_by_size=False
):
    """
    Implements the linear conjugate gradients method for (approximately) solving systems of the form

        lhs result = rhs

    for positive definite and symmetric matrices.

    Args:
      - matmul_closure - a function which performs a left matrix multiplication with lhs_mat
      - rhs - the right-hand side of the equation
      - n_tridiag - returns a tridiagonalization of the first n_tridiag columns of rhs
      - tolerance - stop the solve when the max residual is less than this
      - eps - noise to add to prevent division by zero
      - stop_updating_after - will stop updating a vector after this residual norm is reached
      - max_iter - the maximum number of CG iterations
      - max_tridiag_iter - the maximum size of the tridiagonalization matrix
      - initial_guess - an initial guess at the solution `result`
      - precondition_closure - a functions which left-preconditions a supplied vector

    Returns:
      result - a solution to the system (if n_tridiag is 0)
      result, tridiags - a solution to the system, and corresponding tridiagonal matrices (if n_tridiag > 0)
    """
    # Unsqueeze, if necesasry
    is_vector = rhs.ndimension() == 1
    if is_vector:
        rhs = rhs.unsqueeze(-1)

    # Some default arguments
    if max_iter is None:
        max_iter = rhs.shape[-1]*5
        
    if initial_guess is None:
        initial_guess = torch.zeros_like(rhs)

    if preconditioner is None:
        preconditioner = _default_preconditioner
        precond = False
    else:
        precond = True

    # If we are running m CG iterations, we obviously can't get more than m Lanczos coefficients
    #if max_tridiag_iter > max_iter:
    #    raise RuntimeError("Getting a tridiagonalization larger than the number of CG iterations run is not possible!")

    # Check matmul_closure object
    if torch.is_tensor(matmul_closure):
        matmul_closure = matmul_closure.matmul
    elif not callable(matmul_closure):
        raise RuntimeError("matmul_closure must be a tensor, or a callable object!")

    # Get some constants
    batch_shape = rhs.shape[:-2]
    num_rows = rhs.size(-2)
    n_iter = min(max_iter, num_rows) if terminate_cg_by_size else max_iter
    n_tridiag_iter = 0#min(max_tridiag_iter, num_rows)
    eps = torch.tensor(eps, dtype=rhs.dtype, device=rhs.device)

    # Get the norm of the rhs - used for convergence checks
    # Here we're going to make almost-zero norms actually be 1 (so we don't get divide-by-zero issues)
    # But we'll store which norms were actually close to zero
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)

    # Let's normalize. We'll un-normalize afterwards
    rhs = rhs.div(rhs_norm)

    # residual: residual_{0} = b_vec - lhs x_{0}
    residual = rhs - matmul_closure(initial_guess)

    # result <- x_{0}
    result = initial_guess.expand_as(residual).contiguous()

    # Check for NaNs
    if not torch.equal(residual, residual):
        raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

    # Sometime we're lucky and the preconditioner solves the system right away
    # Check for convergence
    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    if has_converged.all() and not n_tridiag:
        n_iter = 0  # Skip the iteration!

    # Otherwise, let's define precond_residual and curr_conjugate_vec
    else:
        # precon_residual{0} = M^-1 residual_{0}
        precond_residual = preconditioner(residual)
        curr_conjugate_vec = precond_residual
        residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

        # Define storage matrices
        mul_storage = torch.empty_like(residual)
        alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
        beta = torch.empty_like(alpha)
        is_zero = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=torch.bool, device=residual.device)

    # Define tridiagonal matrices, if applicable
    if n_tridiag:
        t_mat = torch.zeros(
            n_tridiag_iter, n_tridiag_iter, *batch_shape, n_tridiag, dtype=alpha.dtype, device=alpha.device
        )
        alpha_tridiag_is_zero = torch.empty(*batch_shape, n_tridiag, dtype=bool_compat, device=t_mat.device)
        alpha_reciprocal = torch.empty(*batch_shape, n_tridiag, dtype=t_mat.dtype, device=t_mat.device)
        prev_alpha_reciprocal = torch.empty_like(alpha_reciprocal)
        prev_beta = torch.empty_like(alpha_reciprocal)

    update_tridiag = True
    last_tridiag_iter = 0

    # It's conceivable we reach the tolerance on the last iteration, so can't just check iteration number.
    tolerance_reached = False

    # Start the iteration
    for k in range(n_iter):
        # Get next alpha
        # alpha_{k} = (residual_{k-1}^T precon_residual{k-1}) / (p_vec_{k-1}^T mat p_vec_{k-1})
        mvms = matmul_closure(curr_conjugate_vec)
        if precond:
            torch.mul(curr_conjugate_vec, mvms, out=mul_storage)
            torch.sum(mul_storage, -2, keepdim=True, out=alpha)

            # Do a safe division here
            torch.lt(alpha, eps, out=is_zero)
            alpha.masked_fill_(is_zero, 1)
            torch.div(residual_inner_prod, alpha, out=alpha)
            alpha.masked_fill_(is_zero, 0)

            # We'll cancel out any updates by setting alpha=0 for any vector that has already converged
            alpha.masked_fill_(has_converged, 0)

            # Update residual
            # residual_{k} = residual_{k-1} - alpha_{k} mat p_vec_{k-1}
            residual = torch.addcmul(residual, alpha, mvms, value=-1, out=residual)

            # Update precond_residual
            # precon_residual{k} = M^-1 residual_{k}
            precond_residual = preconditioner(residual)

            _jit_linear_cg_updates(
                result,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                mul_storage,
                is_zero,
                curr_conjugate_vec,
            )
        else:
            _jit_linear_cg_updates_no_precond(
                mvms,
                result,
                has_converged,
                alpha,
                residual_inner_prod,
                eps,
                beta,
                residual,
                precond_residual,
                mul_storage,
                is_zero,
                curr_conjugate_vec,
            )

        torch.norm(residual, 2, dim=-2, keepdim=True, out=residual_norm)
        residual_norm.masked_fill_(rhs_is_zero, 0)
        torch.lt(residual_norm, stop_updating_after, out=has_converged)

        if k >= 10 and bool(residual_norm.mean() < tolerance) and not (n_tridiag and k < n_tridiag_iter):
            tolerance_reached = True
            break

        # Update tridiagonal matrices, if applicable
        if n_tridiag and k < n_tridiag_iter and update_tridiag:
            alpha_tridiag = alpha.squeeze_(-2).narrow(-1, 0, n_tridiag)
            beta_tridiag = beta.squeeze_(-2).narrow(-1, 0, n_tridiag)
            torch.eq(alpha_tridiag, 0, out=alpha_tridiag_is_zero)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 1)
            torch.reciprocal(alpha_tridiag, out=alpha_reciprocal)
            alpha_tridiag.masked_fill_(alpha_tridiag_is_zero, 0)

            if k == 0:
                t_mat[k, k].copy_(alpha_reciprocal)
            else:
                torch.addcmul(alpha_reciprocal, prev_beta, prev_alpha_reciprocal, out=t_mat[k, k])
                torch.mul(prev_beta.sqrt_(), prev_alpha_reciprocal, out=t_mat[k, k - 1])
                t_mat[k - 1, k].copy_(t_mat[k, k - 1])

                if t_mat[k - 1, k].max() < 1e-6:
                    update_tridiag = False

            last_tridiag_iter = k

            prev_alpha_reciprocal.copy_(alpha_reciprocal)
            prev_beta.copy_(beta_tridiag)

    # Un-normalize
    result = result.mul(rhs_norm)

    if not tolerance_reached and n_iter > 0:
        print(
            "CG terminated in {} iterations with average residual norm {}"
            " which is larger than the tolerance of {} specified by"
            " gpytorch.settings.cg_tolerance."
            " If performance is affected, consider raising the maximum number of CG iterations by running code in"
            " a gpytorch.settings.max_cg_iterations(value) context.".format(k + 1, residual_norm.mean(), tolerance)
        )

    if is_vector:
        result = result.squeeze(-1)

    if n_tridiag:
        t_mat = t_mat[: last_tridiag_iter + 1, : last_tridiag_iter + 1]
        return result, t_mat.permute(-1, *range(2, 2 + len(batch_shape)), 0, 1).contiguous()
    else:
        return result