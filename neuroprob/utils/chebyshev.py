import numpy as np



def Chebyshev_coeff(f, n):
    """
    Use the Gauss-Chebyshev zeros of T_N(x) to compute approximate coefficients with cosine transform.
    Note this gives exact f values at x_k, with controlled error inbetween.
    """
    k = np.arange(n+1)
    x = np.cos(np.pi*(k + 0.5)/(n+1))
    F = f(x)
    T = np.empty((n+1, n+1))
    T[0, :] = 1
    T[1, :] = x
    for kk in range(2, n+1):
        T[kk, :] = 2*x*T[kk-1, :] - T[kk-2, :]
    c = np.empty((n+1,))
    
    # zero
    c[0] = 1/(n+1)*F.sum()
    
    # > 0
    c[1:] = 2/(n+1)*(F[None, :]*T[1:, :]).sum(-1)
    
    return c



def matrix_power_basis_coeff(n, c):
    """
    c_j T_j(x) to b_j x^j basis
    c.T B x_powers = sum_j c_j T_j(x)
    """
    B = np.zeros((n+1, n+1))
    B[0, 0] = 1
    B[1, 1] = 1
    for kk in range(2, n+1):
        B[kk, 0] = -B[kk-2, 0]
        B[kk, 1:] = 2*B[kk-1, :-1] - B[kk-2, 1:]
    
    b = (c[:, None]*B).sum(0)
    return b



def eval_Chebyshev(x, c):
    n = c.shape[0] # n + 1 actually
    T = np.empty((n, x.shape[0]))
    T[0, :] = 1
    T[1, :] = x
    for kk in range(2, n):
        T[kk, :] = 2*x*T[kk-1, :] - T[kk-2, :]
        
    return (c[:, None]*T).sum(0)



def eval_Chebyshev_b(x, b):
    n = b.shape[0] # n + 1 actually
    k = np.arange(n)
    P = x[None, :]**k[:, None]
    return (b[:, None]*P).sum(0)
