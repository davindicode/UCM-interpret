import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

import sys
sys.path.append("../../neuroppl/")

import neuroprob as nprb
from neuroprob import utils





### model ###
def sample_F(mapping, likelihood, covariates, MC, F_dims, trials=1, eps=None):
    """
    Sample F from diagonalized variational posterior.
    
    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)
    if mapping.MC_only or eps is not None:
        samples = mapping.sample_F(cov, eps=eps)[:, F_dims, :] # TODO: cov_samples vs ll_samples?
        h = samples.view(-1, trials, *samples.shape[1:])
    else:
        F_mu, F_var = mapping.compute_F(cov)
        h = likelihood.mc_gen(F_mu, F_var, MC, F_dims)

    return h



def posterior_rate(mapping, likelihood, covariates, MC, F_dims, trials=1, percentiles=[0.05, .5, 0.95]):
    """
    Sample F from diagonalized variational posterior.
    
    :returns: F of shape (MCxtrials, outdims, time)
    """
    cov = mapping.to_XZ(covariates, trials)
    if mapping.MC_only:
        F = mapping.sample_F(cov)[:, F_dims, :] # TODO: cov_samples vs ll_samples?
        samples = likelihood.f(F.view(-1, trials, *samples.shape[1:]))
    else:
        F_mu, F_var = mapping.compute_F(cov)
        samples = likelihood.sample_rate(
            F_mu[:, F_dims, :], F_var[:, F_dims, :], trials, MC)
    
    return utils.signal.percentiles_from_samples(samples, percentiles)
    
    
    
def sample_tuning_curves(mapping, likelihood, covariates, MC, F_dims, trials=1):
    """
    """
    cov = mapping.to_XZ(covariates, trials)
    eps = torch.randn((MC*trials, *cov.shape[1:-1]), 
                      dtype=mapping.tensor_type, device=mapping.dummy.device)
    #mapping.jitter = 1e-4
    samples = mapping.sample_F(cov, eps)
    T = samples.view(-1, trials, *samples.shape[1:])

    return T



def sample_Y(mapping, likelihood, covariates, trials, MC=1):
    """
    Use the posterior mean rates. Sampling gives np.ndarray
    """
    cov = mapping.to_XZ(covariates, trials)
    
    with torch.no_grad():
            
        F_mu, F_var = mapping.compute_F(cov)
        rate = likelihood.sample_rate(F_mu, F_var, trials, MC) # MC, trials, neuron, time

        rate = rate.mean(0).cpu().numpy()
        syn_train = likelihood.sample(rate, XZ=cov)
        
    return syn_train




### UCM ###
def compute_P(full_model, covariates, show_neuron, MC=1000, trials=1):
    """
    Compute predictive count distribution given X.
    """
    F_dims = full_model.likelihood._neuron_to_F(show_neuron)
    h = sample_F(full_model.mapping, full_model.likelihood, covariates, MC, F_dims, 
                 trials=trials)
    logp = full_model.likelihood.get_logp(h, show_neuron).data # samples, N, time, K
    
    P_mc = torch.exp(logp)
    return P_mc    




def marginalized_P(full_model, eval_points, eval_dims, rcov, bs, use_neuron, MC=100, skip=1):
    """
    Marginalize over the behaviour p(X) for X not evaluated over.

    :param list eval_points: list of ndarrays of values that you want to compute the marginal SCD at
    :param list eval_dims: the dimensions that are not marginalized evaluated at eval_points
    :param list rcov: list of covariate time series
    :param int bs: batch size
    :param list use_neuron: list of neurons used
    :param int skip: only take every skip time points of the behaviour time series for marginalisation
    """
    rcov = [rc[::skip] for rc in rcov] # set dilution
    animal_T = rcov[0].shape[0]
    Ep = eval_points[0].shape[0]
    tot_len = Ep*animal_T
    
    covariates = []
    k = 0
    for d, rc in enumerate(rcov):
        if d in eval_dims:
            covariates.append(
                torch.repeat_interleave(eval_points[k], animal_T)
            )
            k += 1
        else:
            covariates.append(rc.repeat(Ep))
    
    km = full_model.likelihood.K+1
    P_tot = torch.empty((MC, len(use_neuron), Ep, km), dtype=torch.float)
    batches = int(np.ceil(animal_T / bs))
    for e in range(Ep):
        print(e)
        P_ = torch.empty((MC, len(use_neuron), animal_T, km), dtype=torch.float)
        for b in range(batches):
            bcov = [c[e*animal_T:(e+1)*animal_T][b*bs:(b+1)*bs] for c in covariates]
            P_mc = compute_P(full_model, bcov, use_neuron, MC=MC).cpu()
            P_[..., b*bs:(b+1)*bs, :] = P_mc
            
        P_tot[..., e, :] = P_.mean(-2)
        
    return P_tot




def marginalized_T(full_model, T_funcs, T_num, eval_points, eval_dims, rcov, bs, use_neuron, MC=100, skip=1):
    """
    Marginalize over the behaviour p(X) for X not evaluated over.
    """
    rcov = [rc[::skip] for rc in rcov] # set dilution
    animal_T = rcov[0].shape[0]
    Ep = eval_points[0].shape[0]
    tot_len = Ep*animal_T
    
    covariates = []
    k = 0
    for d, rc in enumerate(rcov):
        if d in eval_dims:
            covariates.append(
                torch.repeat_interleave(eval_points[k], animal_T)
            )
            k += 1
        else:
            covariates.append(rc.repeat(Ep))

    T_tot = torch.empty((MC, len(use_neuron), Ep, T_num), dtype=torch.float)
    batches = int(np.ceil(animal_T / bs))
    for e in range(Ep):
        print(e)
        T_ = torch.empty((MC, len(use_neuron), animal_T, T_num), dtype=torch.float)
        for b in range(batches):
            bcov = [c[e*animal_T:(e+1)*animal_T][b*bs:(b+1)*bs] for c in covariates]
            P_mc = compute_P(full_model, bcov, use_neuron, MC=MC).cpu()
            T_[..., b*bs:(b+1)*bs, :] = T_funcs(P_mc)
            
        T_tot[..., e, :] = T_.mean(-2)
        
    return T_tot



### tuning ###
def T_funcs(P):
    """
    Outputs (mean, var, FF) in last dimension
    """
    x_counts = torch.arange(P.shape[-1])
    x_count_ = x_counts[None, None, None, :]

    mu_ = (x_count_*P).sum(-1) # mc, N, T
    var_ = ((x_count_**2*P).sum(-1)-mu_**2) # mc, N, T
    FF_ = (var_/(mu_+1e-12)) # mc, N, T
    
    return torch.stack((mu_, var_, FF_), dim=-1)



def TI(T):
    """
    tuning indices
    """
    return torch.abs((T.max(dim=-1)[0] - T.min(dim=-1)[0]) / (T.max(dim=-1)[0] + T.min(dim=-1)[0]))



def R2(T_marg, T_full):
    """
    explained variance, R squared
    total variance decomposition of posterior mean values
    """
    return 1 - ((T_full-T_marg)**2).mean(-1)/T_full.var(-1)


def RV(T_marg, T_full):
    """
    relative variance
    """
    return T_marg.var(-1)/T_full.var(-1)



def rTI(T, T_std):
    """
    tuning indices
    """
    return (T.max(dim=-1)[0] - T.min(dim=-1)[0]) / (T_std.mean(-1))

    

def rR2(T_marg, T_full):
    """
    explained variance, R squared
    total variance decomposition of posterior mean values
    """
    return 1 - ((T_full-T_marg)**2).mean(-1) / \
           (T_full.var(-1).mean(0) + T_full.mean(-1).var(0))

    
def MINE():
    return



### stats ###
def ind_to_pair(ind, N):
    a = ind
    k = 1
    while a >= 0:
        a -= (N-k)
        k += 1
        
    n = k-1
    m = N-n + a
    return n-1, m



def get_q_Z(P, spike_binned, deq_noise=None):  
    if deq_noise is None:
        deq_noise = np.random.uniform(size=spike_binned.shape)
    else:
        deq_noise = 0

    cumP = np.cumsum(P, axis=-1) # T, K
    tt = np.arange(spike_binned.shape[0])
    quantiles = cumP[tt, spike_binned.astype(int)] - P[tt, spike_binned.astype(int)]*deq_noise
    Z = utils.stats.q_to_Z(quantiles)
    return quantiles, Z





def compute_count_stats(modelfit, spktrain, behav_list, neuron, traj_len=None, traj_spikes=None,
                        start=0, T=100000, bs=5000, n_samp=1000):
    """
    Compute the dispersion statistics for the count model
    
    :param string mode: *single* mode refers to computing separate single neuron quantities, *population* mode 
                        refers to computing over a population indicated by neurons, *peer* mode involves the 
                        peer predictability i.e. conditioning on all other neurons given a subset
    """
    mapping = modelfit.mapping
    likelihood = modelfit.likelihood
    tbin = modelfit.likelihood.tbin
    
    N = int(np.ceil(T/bs))
    rate_model = []
    shape_model = []
    spktrain = spktrain[:, start:start+T]
    behav_list = [b[start:start+T] for b in behav_list]
    
    for k in range(N):
        covariates_ = [torchb[k*bs:(k+1)*bs] for b in behav_list]
        ospktrain = spktrain[None, ...]

        rate = posterior_rate(
            mapping, likelihood, covariates, MC, F_dims, trials=1, percentiles=[.5]
        )#glm.mapping.eval_rate(covariates_, neuron, n_samp=1000)
        rate_model += [rate[0, ...]]
        
        if likelihood.dispersion_mapping is not None:
            cov = mapping.to_XZ(covariates_, trials=1)
            disp = likelihood.sample_dispersion(cov, n_samp, neuron)
            shape_model += [disp[0, ...]]
            
    rate_model = np.concatenate(rate_model, axis=1)
    if count_model and glm.likelihood.dispersion_mapping is not None:
        shape_model = np.concatenate(shape_model, axis=1)
    
    if type(likelihood) == nprb.likelihoods.Poisson:
        shape_model = None
        f_p = lambda c, avg, shape, t: utils.stats.poiss_count_prob(c, avg, shape, t)
        
    elif type(likelihood) == nprb.likelihoods.Negative_binomial:
        shape_model = glm.likelihood.r_inv.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.nb_count_prob(c, avg, shape, t)
        
    elif type(likelihood) == nprb.likelihoods.COM_Poisson:
        shape_model = glm.likelihood.nu.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.cmp_count_prob(c, avg, shape, t)
        
    elif type(likelihood) == nprb.likelihoods.ZI_Poisson:
        shape_model = glm.likelihood.alpha.data.cpu().numpy()[:, None].repeat(T, axis=-1)
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)
        
    elif type(likelihood) == nprb.likelihoods.hNegative_binomial:
        f_p = lambda c, avg, shape, t: utils.stats.nb_count_prob(c, avg, shape, t)
        
    elif type(likelihood) == nprb.likelihoods.hCOM_Poisson:
        f_p = lambda c, avg, shape, t: utils.stats.cmp_count_prob(c, avg, shape, t)
        
    elif type(likelihood) == nprb.likelihoods.hZI_Poisson:
        f_p = lambda c, avg, shape, t: utils.stats.zip_count_prob(c, avg, shape, t)
        
    else:
        raise ValueError
        
    m_f = lambda x: x

    if shape_model is not None:
        assert traj_len == 1
    if traj_len is not None:
        traj_lens = (T // traj_len) * [traj_len]
        
    q_ = []
    for k, ne in enumerate(neuron):
        if traj_spikes is not None:
            avg_spikecnt = np.cumsum(rate_model[k]*tbin)
            nc = 1
            traj_len = 0
            for tt in range(T):
                if avg_spikecnt >= traj_spikes*nc:
                    nc += 1
                    traj_lens.append(traj_len)
                    traj_len = 0
                    continue
                traj_len += 1
                
        if shape_model is not None:
            sh = shape_model[k]
            spktr = spktrain[ne]
            rm = rate_model[k]
        else:
            sh = None
            spktr = []
            rm = []
            start = np.cumsum(traj_lens)
            for tt, traj_len in enumerate(traj_lens):
                spktr.append(spktrain[ne][start[tt]:start[tt]+traj_len].sum())
                rm.append(rate_model[k][start[tt]:start[tt]+traj_len].sum())
            spktr = np.array(spktr)
            rm = np.array(rm)
                    
        q_.append(utils.stats.count_KS_method(f_p, m_f, tbin, spktr, rm, shape=sh))

    return q_




def compute_isi_stats(mapping, likelihood, tbin, spktrain, behav_list, neuron, traj_len=None, traj_spikes=None,
                        start=0, T=100000, bs=5000):
    return




def compute_IPP_stats(mapping, likelihood, tbin, spktrain, behav_list, neuron, traj_len=None, traj_spikes=None,
                        start=0, T=100000, bs=5000):
    return




### spike conversions ###
def spikeinds_from_count(counts, bin_elements):
    """
    convert to spike timings
    
    returns: list of list of array of spike time indices
    """ 
    spikes = np.zeros(counts.shape + (bin_elements,))

    spikeinds = []
    trs, neurons, T = counts.shape
    for tr in range(trs):
        spikeinds_ = []
        for n in range(neurons):
            spikeinds__ = []
            cur_steps = 0
            for t in range(T):
                c = int(counts[tr, n, t])
                #spikeinds__.append(np.sort(random.sample(range(bin_elements), c)) + cur_steps) without replacement
                spikeinds__.append(np.sort(random.choices(range(bin_elements), k=c)) + cur_steps) # with replacement
                cur_steps += bin_elements

            spikeinds_.append(np.concatenate(spikeinds__))
        spikeinds.append(spikeinds_)
            
    return spikeinds
            
    
    
def rebin_spikeinds(spikeinds, bin_sizes, dt, behav_tuple, average_behav=False):
    """
    rebin data
    """
    datasets = []
    for bin_size in bin_sizes:
        tbin, resamples, rc_t, rbehav_tuple = utils.neural.bin_data(
            bin_size, dt, spikeinds, track_samples*C, behav_tuple, 
            average_behav=average_behav, binned=False
        )

        datasets.append((tbin, resamples, rc_t, rbehav_tuple))
    return datasets
    #res_ind = [] # spike times
    #for r in res:
    #    res_ind.append(utils.neural.binned_to_indices(r))





### cross validation ###
def RG_pred_ll(model, validation_set, neuron_group=None, ll_mode='GH', ll_samples=100, cov_samples=1, 
               beta=1.0, IW=False):
    """
    Compute the predictive log likelihood (ELBO).
    """
    vcov, vtrain, vbatch_info = validation_set
    time_steps = vtrain.shape[-1]
    print('Data segment timesteps: {}'.format(time_steps))
    
    model.input_group.set_XZ(vcov, time_steps, batch_info=vbatch_info)
    model.likelihood.set_Y(vtrain, batch_info=vbatch_info)
    model.validate_model(likelihood_set=True)
    
    # batching
    pll = []
    for b in range(model.input_group.batches):
        pll.append(-model.objective(b, cov_samples=cov_samples, ll_mode=ll_mode, neuron=neuron_group, 
                                    beta=beta, ll_samples=ll_samples, importance_weighted=IW).item())
    
    return np.array(pll).mean()
    
    

def LVM_pred_ll(model, validation_set, f_neuron, v_neuron, 
                eval_cov_MC=1, eval_ll_MC=100, eval_ll_mode='GH', annealing=lambda x: 1.0, #min(1.0, 0.002*x)
                cov_MC=16, ll_MC=1, ll_mode='MC', beta=1.0, IW=False, max_iters=3000):
    """
    Compute the predictive log likelihood (ELBO).
    """
    vcov, vtrain, vbatch_info = validation_set
    time_steps = vtrain.shape[-1]
    print('Data segment timesteps: {}'.format(time_steps))
    
    model.input_group.set_XZ(vcov, time_steps, batch_info=vbatch_info)
    model.likelihood.set_Y(vtrain, batch_info=vbatch_info)
    model.validate_model(likelihood_set=True)
    
    # fit
    sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: 0.9)
    opt_tuple = (optim.Adam, 100, sch)
    opt_lr_dict = {'default': 0}
    for z_dim in model.input_group.latent_dims:
        opt_lr_dict['input_group.input_{}.variational.mu'.format(z_dim)] = 1e-2
        opt_lr_dict['input_group.input_{}.variational.finv_std'.format(z_dim)] = 1e-3

    model.set_optimizers(opt_tuple, opt_lr_dict)#, nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

    
    losses = model.fit(max_iters, neuron=f_neuron, loss_margin=-1e0, margin_epochs=100, ll_mode=ll_mode, 
                       kl_anneal_func=annealing, cov_samples=cov_MC, ll_samples=ll_MC)

    pll = []
    for b in range(model.input_group.batches):
        pll.append(-model.objective(b, neuron=v_neuron, cov_samples=eval_cov_MC, ll_mode=eval_ll_mode, 
                                    beta=beta, ll_samples=eval_ll_MC, importance_weighted=IW).item())
        
    return np.array(pll).mean(), losses
