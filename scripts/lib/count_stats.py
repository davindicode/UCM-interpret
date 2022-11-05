import torch
import torch.optim as optim

import numpy as np
import scipy
import matplotlib.pyplot as plt


import sys
sys.path.append("../../neuroppl/")

import neuroprob as nprb
from neuroprob import utils

import lib



def variability_stats(modelfit, gp_dev='cpu', MC=100, bs=10000, jitter=1e-5):
    """
    full input data
    """
    units_used = modelfit.likelihood.neurons
    
    P_full = []
    with torch.no_grad():
        for b in range(modelfit.input_group.batches):
            XZ = modelfit.input_group.sample_XZ(b, 1, None, False)[0]
            P_mc = lib.helper.compute_P(modelfit, XZ, list(range(units_used)), MC=MC).cpu()
            P_full.append(P_mc)

    P_full = torch.cat(P_full, dim=-2)


    ### variability ###
    T_full = lib.helper.T_funcs(P_full)
    mu_full, var_full, FF_full = T_full[..., 0], T_full[..., 1], T_full[..., 2]

    # posterior mean stats
    A = mu_full.mean(0)
    B = FF_full.mean(0)
    C = var_full.mean(0)
    resamples = A.shape[-1]

     
    # linear variance rate
    a, b = utils.signal.linear_regression(A, C)

    B_fit = a[:, None] + b[:, None]/(A+1e-12)
    C_fit = a[:, None]*A + b[:, None]
    # R^2 of fit
    R2_ff = 1 - (B-B_fit).var(-1) / B.var(-1)
    R2_var = 1 - (C-C_fit).var(-1) / C.var(-1)

    linvar_tuple = (a, b, R2_ff, R2_var)


    # constant FF
    mff = B.mean(-1) # R^2 by definition 0
    C_fit = mff[:, None]*A
    R2_var = 1 - (C-C_fit).var(-1) / C.var(-1)
    constFF_tuple = (mff, R2_var)


    # linear fits (doubly stochastic models, refractory models)
    a, b = utils.signal.linear_regression(A, B)

    B_fit = a[:, None]*A + b[:, None]
    C_fit = a[:, None]*A**2 + b[:, None]*A
    # R^2 of fit
    R2_ff = 1 - (B-B_fit).var(-1) / B.var(-1)
    R2_var = 1 - (C-C_fit).var(-1) / C.var(-1)

    linff_tuple = (a, b, R2_ff, R2_var)
    
    
    # nonparametric fits
    y_dims = units_used
    covariates = []
    for ne in range(y_dims):
        covariates += [np.linspace(0, A[ne, :].max()*1.01, 100)]
    covariates = torch.tensor(covariates)
        
    np_tuple = (covariates,)
    for yd in [B, C]:
        v = 1.*torch.ones(y_dims)
        l = 1.*torch.ones(1, y_dims)

        constraints = []
        krn_1 = nprb.kernels.kernel.Constant(variance=v, tensor_type=torch.float)
        krn_2 = nprb.kernels.kernel.SquaredExponential(
            input_dims=1, lengthscale=l, \
            topology='torus', f='softplus', \
            track_dims=[0], tensor_type=torch.float
        )

        kernel = nprb.kernels.kernel.Product(krn_1, krn_2)

        num_induc = 8
        Xu = []
        for ne in range(y_dims):
            Xu += [np.linspace(0, A[ne, :].max(), num_induc)]
        Xu = torch.tensor(Xu)[..., None]
        inducing_points = nprb.kernels.kernel.inducing_points(y_dims, Xu, constraints)

        input_data = [A[None, :, :, None]] # tr, N, T, D

        # mapping
        in_dims = Xu.shape[-1]

        gp = nprb.mappings.GP.SVGP(
            in_dims, y_dims, kernel, inducing_points=inducing_points, 
            whiten=True, jitter=jitter, mean=torch.zeros(y_dims), learn_mean=True
        )

        ### inputs and likelihood ###
        input_group = nprb.inference.input_group()
        input_group.set_XZ(input_data, resamples, batch_info=bs)

        likelihood = nprb.likelihoods.Gaussian(y_dims, 'exp', log_var=torch.zeros(y_dims))
        likelihood.set_Y(yd, batch_info=bs) 

        gpr = nprb.inference.VI_optimized(input_group, gp, likelihood)
        gpr.to(gp_dev)

        # fitting
        sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
        opt_tuple = (optim.Adam, 100, sch)
        opt_lr_dict = {'default': 5*1e-3}

        gpr.set_optimizers(opt_tuple, opt_lr_dict)

        annealing = lambda x: 1.0
        losses = gpr.fit(3000, loss_margin=0.0, margin_epochs=100, kl_anneal_func=annealing, 
                         cov_samples=1, ll_samples=10, ll_mode='MC')

        plt.figure()
        plt.plot(losses)
        plt.xlabel('epoch')
        plt.ylabel('NLL per time sample')
        plt.show()


        with torch.no_grad():
            lw, mn, up = lib.helper.posterior_rate(gp, likelihood, covariates[None, ..., None], 10000, 
                                                   F_dims=list(range(units_used)), percentiles=[0.05, .5, 0.95])
            lw = lw[0, ...].cpu()
            mn = mn[0, ...].cpu()
            up = up[0, ...].cpu()

        _fit_ = []
        bs = 100
        batches = int(np.ceil(A.shape[-1] / bs))
        with torch.no_grad():
            for b in range(batches):
                _fit_.append(
                    lib.helper.posterior_rate(gp, likelihood, [A[None, :, b*bs:(b+1)*bs, None]], 10000, 
                                              F_dims=list(range(units_used)), percentiles=[.5])[0][0, ...].cpu()
                )

        _fit = torch.cat(_fit_, dim=-1)
        R2 = 1 - (yd-_fit).var(-1) / yd.var(-1)

        np_tuple += (lw, mn, up, R2)


    ### collect ###
    variability_stats = (
        T_full.permute(0, 1, -1, -2), 
        linvar_tuple, constFF_tuple, linff_tuple, \
        np_tuple
    )
    return variability_stats
    
    
    
    
def marginal_stats(modelfit, rcov_used, T_full, dimx_list, ang_dims=[], MC=100, skip=10, batchsize=10000, 
                   grid_size_pos=(50, 40), grid_size_1d=101):
    """
    marginalized data
    2D tuning curves have image convention T(y, x) (row-column)
    """
    units_used = modelfit.likelihood.neurons

    marginal_stats = []
    for dimx in dimx_list:
        if len(dimx) == 1:
            if dimx in ang_dims:
                cov_list = [torch.linspace(0, 2*np.pi, grid_size_1d)]
            else:
                cov_list = [torch.linspace(rcov_used[dimx[0]].min(), rcov_used[dimx[0]].max(), grid_size_1d)]

        elif len(dimx) == 2:
            A, B = grid_size_pos
            cov_list = [
                torch.linspace(rcov_used[dimx[0]].min(), rcov_used[dimx[0]].max(), A)[None, :].repeat(B, 1).flatten(), 
                torch.linspace(rcov_used[dimx[1]].min(), rcov_used[dimx[1]].max(), B)[:, None].repeat(1, A).flatten()
            ]

        with torch.no_grad():
            #[torch.linspace(-2*np.pi*(10/100.), 2*np.pi*(110./100.), 121) # approximate periodic boundaries
            P_marg = lib.helper.marginalized_P(
                modelfit, cov_list, dimx, rcov_used, batchsize, 
                list(range(units_used)), MC=MC, skip=skip
            ) # mc, N, T, count

            T_marg = lib.helper.marginalized_T(
                modelfit, lib.helper.T_funcs, 3, cov_list, dimx, rcov_used, batchsize, 
                list(range(units_used)), MC=MC, skip=skip
            ).permute(0, 1, -1, -2) # mc, N, moment_dim, T

        T_marg_d = lib.helper.T_funcs(P_marg).permute(0, 1, -1, -2) # mc, N, moment_dim, T


        ### tuning curves ###
        #mu_marg, var_marg, FF_marg = T_marg[..., 0], T_marg[..., 1], T_marg[..., 2]
        #mu_marg_d, var_marg_d, FF_marg_d = T_marg_d[..., 0], T_marg_d[..., 1], T_marg_d[..., 2]


        ### measures ###
        a, b, c, d = T_marg.shape # mc, N, moment_dim, T

        if len(dimx) == 1:
            Ns = cov_list[0].shape[0]

            T_marg_x = utils.signal.cubic_interpolation(
                cov_list[0], T_marg.mean(0).reshape(-1, Ns), rcov_used[dimx[0]].float(), integrate=False
            ).view(b, c, -1)

            T_marg_d_x = utils.signal.cubic_interpolation(
                cov_list[0], T_marg_d.mean(0).reshape(-1, Ns), rcov_used[dimx[0]].float(), integrate=False
            ).view(b, c, -1)

        elif len(dimx) == 2: # note it is represented as flat vector N_x*N_y
            Ns_x, Ns_y = grid_size_pos
            cl_x = cov_list[0][:Ns_x].numpy()
            cl_y = cov_list[1][::Ns_x].numpy()
            #xx, yy = np.meshgrid(cl_x, cl_y)
            
            x = rcov_used[dimx[0]].numpy()
            y = rcov_used[dimx[1]].numpy()
            #inp = np.stack((x, y))

            z_T = T_marg.numpy().reshape(-1, Ns_y, Ns_x)
            z_T_d = T_marg_d.numpy().reshape(-1, Ns_y, Ns_x)
            
            #steps = x.shape[0]
            #bs = 1000
            #batches = int(np.ceil(steps/bs))
            
            T_marg_x_ = []
            T_marg_d_x_ = []
            for ne in range(z_T.shape[0]):
                #RegularGridInterpolator((cl_y, cl_x), z_T[ne]) # linear
                f = scipy.interpolate.RectBivariateSpline(cl_y, cl_x, z_T[ne])
                T_marg_x_.append(torch.from_numpy(f(x, y, grid=False)).float().flatten())
                
                f = scipy.interpolate.RectBivariateSpline(cl_y, cl_x, z_T_d[ne])
                T_marg_d_x_.append(torch.from_numpy(f(x, y, grid=False)).float().flatten())

                """
                T_marg_x = utils.signal.bilinear_interpolation(
                    cov_list[0], cov_list[1], T_marg.permute(0, 1, -1, -2).reshape(-1, Ns_x, Ns_y), 
                    rcov_used[dimx[0]].float(), rcov_used[dimx[1]].float(), integrate=False
                ).view(a, b, d, -1).permute(0, 1, -1, -2)

                T_marg_d_x = utils.signal.bilinear_interpolation(
                    cov_list[0], cov_list[1], T_marg_d.permute(0, 1, -1, -2).reshape(-1, Ns_x, Ns_y), 
                    rcov_used[dimx[0]].float(), integrate=False
                ).view(a, b, d, -1).permute(0, 1, -1, -2)
                """
            T_marg_x = torch.stack(T_marg_x_, dim=0).view(a, b, c, -1)
            T_marg_d_x = torch.stack(T_marg_d_x_, dim=0).view(a, b, c, -1)
            
        #mu_full, var_full, FF_full = T_full[..., 0], T_full[..., 1], T_full[..., 2]
        #mu_marg_x, var_marg_x, FF_marg_x = T_marg_x[..., 0], T_marg_x[..., 1], T_marg_x[..., 2]
        #mu_marg_d_x, var_marg_d_x, FF_marg_d_x = T_marg_d_x[..., 0], T_marg_d_x[..., 1], T_marg_d_x[..., 2]
        T_full_pm = T_full
        
        TI_d = lib.helper.TI(T_marg_d)
        R2_d = lib.helper.R2(T_marg_d_x, T_full_pm)
        RV_d = lib.helper.RV(T_marg_d_x, T_full_pm)

        TI = lib.helper.TI(T_marg)
        R2 = lib.helper.R2(T_marg_x, T_full_pm)
        RV = lib.helper.RV(T_marg_x, T_full_pm)


        marginal_stats.append(
            (cov_list, T_marg, T_marg_d, 
             TI_d, R2_d, RV_d, TI, R2, RV)
        )
        
    return marginal_stats