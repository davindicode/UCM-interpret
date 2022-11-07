import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import argparse

import pickle
import os
import sys
sys.path.append("..")

import neuroprob as nprb
from neuroprob import utils
from neuroprob import kernels



### GP ###
def create_kernel(kernel_tuples, kern_f, tensor_type):
    """
    Helper function for creating kernel triplet tuple
    """
    track_dims = 0
    kernelobj = 0

    constraints = []
    for k, k_tuple in enumerate(kernel_tuples):

        if k_tuple[0] is not None:

            if k_tuple[0] == 'variance':
                krn = kernels.kernel.Constant(variance=k_tuple[1], tensor_type=tensor_type)

            else:
                kernel_type = k_tuple[0]
                topology = k_tuple[1]
                lengthscales = k_tuple[2]

                if topology == 'sphere':
                    constraints += [(track_dims, track_dims+len(lengthscales), 'sphere'),]

                act = []
                for _ in lengthscales:
                    act += [track_dims]
                    track_dims += 1

                if kernel_type == 'SE':
                    krn = kernels.kernel.SquaredExponential(
                        input_dims=len(lengthscales), lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'DSE':
                    if topology != 'euclid':
                        raise ValueError('Topology must be euclid')
                    lengthscale_beta = k_tuple[3]
                    beta = k_tuple[4]
                    krn = kernels.kernel.DSE(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        lengthscale_beta=lengthscale_beta, \
                        beta=beta, \
                        track_dims=act, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'OU':
                    krn = kernels.kernel.Exponential(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'RQ':
                    scale_mixture = k_tuple[3]
                    krn = kernels.kernel.RationalQuadratic(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        scale_mixture=scale_mixture, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'Matern32':
                    krn = kernels.kernel.Matern32(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'Matern52':
                    krn = kernels.kernel.Matern52(
                        input_dims=len(lengthscales), \
                        lengthscale=lengthscales, \
                        track_dims=act, topology=topology, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                elif kernel_type == 'linear':
                    if topology != 'euclid':
                        raise ValueError('Topology must be euclid')
                    krn = kernels.kernel.Linear(
                        input_dims=len(lengthscales), \
                        track_dims=act, f=kern_f
                    )
                    
                elif kernel_type == 'polynomial':
                    if topology != 'euclid':
                        raise ValueError('Topology must be euclid')
                    degree = k_tuple[3]
                    krn = kernels.kernel.Polynomial(
                        input_dims=len(lengthscales), \
                        bias=lengthscales, \
                        degree=degree, track_dims=act, f=kern_f, \
                        tensor_type=tensor_type
                    )
                    
                else:
                    raise NotImplementedError('Kernel type is not supported.')

            kernelobj = kernels.kernel.Product(kernelobj, krn) if kernelobj != 0 else krn

        else:
            track_dims += 1

    return kernelobj, constraints



def latent_kernel(z_mode, num_induc, out_dims):
    """
    """
    z_mode_comps = z_mode.split('-')
    
    ind_list = []
    kernel_tuples = []
    
    l_one = np.ones(out_dims)
    
    for zc in z_mode_comps:
        if zc[:1] == 'R':
            dz = int(zc[1:]) 
            for h in range(dz):
                ind_list += [np.random.randn(num_induc)]
            kernel_tuples += [('SE', 'euclid', torch.tensor([l_one]*dz))]

        elif zc != '':
            raise ValueError('Invalid latent covariate type')
        
    return kernel_tuples, ind_list


### model components ###
def get_basis(basis_mode='ew'):
    
    if basis_mode == 'id':
        basis = (lambda x: x,)
    
    elif basis_mode == 'ew': # element-wise
        basis = (lambda x: x, lambda x: torch.exp(x))
        
    elif basis_mode == 'eq': # element-wise exp-quadratic
        basis = (lambda x: x, lambda x: x**2, lambda x: torch.exp(x))
        
    elif basis_mode == 'ec': # element-wise exp-cubic
        basis = (lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: torch.exp(x))
        
    elif basis_mode == 'qd': # exp and full quadratic
        def mix(x):
            C = x.shape[-1]
            out = torch.empty((*x.shape[:-1], C*(C-1)//2), dtype=x.dtype).to(x.device)
            k = 0
            for c in range(1, C):
                for c_ in range(c):
                    out[..., k] = x[..., c]*x[..., c_]
                    k += 1
                
            return out  # shape (..., C*(C-1)/2)
        
        basis = (lambda x: x, lambda x: x**2, lambda x: torch.exp(x), lambda x: mix(x))
    
    else:
        raise ValueError('Invalid basis expansion')
    
    return basis



class net(nn.Module):
    def __init__(self, C, basis, max_count, channels, shared_W=False):
        super().__init__()
        self.basis = basis
        self.C = C
        expand_C = torch.cat([f_(torch.ones(1, self.C)) for f_ in self.basis], dim=-1).shape[-1]
        
        mnet = nprb.neural_nets.networks.Parallel_MLP(
            [], expand_C, (max_count+1), channels, shared_W=shared_W, out=None
        )  # single linear mapping
        self.add_module('mnet', mnet)
        
        
    def forward(self, input, neuron):
        """
        :param torch.tensor input: input of shape (samplesxtime, in_dimsxchannels)
        """
        input = input.view(input.shape[0], -1, self.C)
        input = torch.cat([f_(input) for f_ in self.basis], dim=-1)
        out = self.mnet(input, neuron)
        return out.view(out.shape[0], -1) # t, NxK



def latent_objects(z_mode, d_x, timesamples, tensor_type):
    """
    """
    z_mode_comps = z_mode.split('-')
    
    tot_d_z, latents = 0, []
    for zc in z_mode_comps:
        d_z = 0
        
        if zc[:1] == 'R':
            d_z = int(zc[1:])

            if d_z == 1:
                p = nprb.inputs.priors.AR1(
                    torch.tensor(0.), torch.tensor(4.), 1, tensor_type=tensor_type)
            else:
                p = nprb.inputs.priors.AR1(
                    torch.tensor([0.]*d_z), torch.tensor([4.]*d_z), d_z, tensor_type=tensor_type)

            v = nprb.inputs.variational.IndNormal(
                torch.rand(timesamples, d_z)*0.1, torch.ones((timesamples, d_z))*0.01, 
                'euclid', d_z, tensor_type=tensor_type)

            latents += [nprb.inference.prior_variational_pair(d_z, p, v)]

        elif zc[:1] == 'T':
            d_z = int(zc[1:])

            if d_z == 1:
                p = nprb.inputs.priors.dAR1(
                    torch.tensor(0.), torch.tensor(4.0), 'torus', 1, tensor_type=tensor_type)
            else:
                p = nprb.inputs.priors.dAR1(
                    torch.tensor([0.]*d_z), torch.tensor([4.]*d_z), 'torus', d_z, tensor_type=tensor_type)

            v = nprb.inputs.variational.IndNormal(
                torch.rand(timesamples, 1)*2*np.pi, torch.ones((timesamples, 1))*0.1, 
                'torus', d_z, tensor_type=tensor_type)

            latents += [nprb.inference.prior_variational_pair(_z, p, v)]

        elif zc != '':
            raise ValueError('Invalid latent covariate type')
            
        tot_d_z += d_z

    return latents, tot_d_z
    
    
def inputs_used(model_dict, covariates, batch_info):
    """
    Create the used covariates list.
    """
    x_mode, z_mode, tensor_type = model_dict['x_mode'], model_dict['z_mode'], model_dict['tensor_type']
    x_mode_comps = x_mode.split('-')
    
    input_data = []
    for xc in x_mode_comps:
        if xc == '':
            continue
        input_data.append(torch.from_numpy(covariates[xc]))
        
    d_x = len(input_data)
    
    timesamples = list(covariates.values())[0].shape[0]
    latents, d_z = latent_objects(z_mode, d_x, timesamples, tensor_type)
    input_data += latents
    return input_data, d_x, d_z
    
    

def get_likelihood(model_dict, enc_used):
    """
    Create the likelihood object.
    """
    ll_mode, tensor_type = model_dict['ll_mode'], model_dict['tensor_type']
    ll_mode_comps = ll_mode.split('-')
    C = int(ll_mode_comps[2]) if ll_mode_comps[0] == 'U' else 1
    
    max_count, neurons, tbin = model_dict['max_count'], model_dict['neurons'], model_dict['tbin']
    inner_dims = model_dict['map_outdims']
    
    if ll_mode[0] == 'h':
        hgp = enc_used(model_dict, cov, learn_mean=False)
    
    if ll_mode_comps[0] == 'U':
        inv_link = 'identity'
    elif ll_mode == 'IBP':
        inv_link = lambda x: torch.sigmoid(x)/tbin
    elif ll_mode_comps[-1] == 'exp':
        inv_link = 'exp'
    elif ll_mode_comps[-1] == 'spl':
        inv_link = 'softplus'
    else:
        raise ValueError('Likelihood inverse link function not defined')
            
    if ll_mode_comps[0] == 'IBP':
        likelihood = nprb.likelihoods.Bernoulli(tbin, inner_dims, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] == 'IP':
        likelihood = nprb.likelihoods.Poisson(tbin, inner_dims, inv_link, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] == 'ZIP':
        alpha = .1*torch.ones(inner_dims)
        likelihood = nprb.likelihoods.ZI_Poisson(tbin, inner_dims, inv_link, alpha, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] =='hZIP':
        likelihood = nprb.likelihoods.hZI_Poisson(tbin, inner_dims, inv_link, hgp, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] == 'NB':
        r_inv = 10.*torch.ones(inner_dims)
        likelihood = nprb.likelihoods.Negative_binomial(tbin, inner_dims, inv_link, r_inv, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] =='hNB':
        likelihood = nprb.likelihoods.hNegative_binomial(tbin, inner_dims, inv_link, hgp, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] == 'CMP':
        J = int(ll_mode_comps[1])
        log_nu = torch.zeros(inner_dims)
        likelihood = nprb.likelihoods.COM_Poisson(tbin, inner_dims, inv_link, log_nu, J=J, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] =='hCMP':
        J = int(ll_mode[4:])
        likelihood = nprb.likelihoods.hCOM_Poisson(tbin, inner_dims, inv_link, hgp, J=J, tensor_type=tensor_type)
        
    elif ll_mode == 'IPP':
        likelihood = nprb.likelihoods.Poisson_pp(tbin, inner_dims, inv_link, tensor_type=tensor_type)
        
    elif ll_mode == 'IG':
        shape = torch.ones(inner_dims)
        likelihood = nprb.likelihoods.Gamma(tbin, inner_dims, inv_link, shape, 
                                            allow_duplicate=True, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] == 'IIG':
        mu_t = torch.ones(inner_dims)
        likelihood = nprb.likelihoods.inv_Gaussian(tbin, inner_dims, inv_link, mu_t, 
                                                   allow_duplicate=True, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] == 'LN':
        sigma_t = torch.ones(inner_dims)
        likelihood = nprb.likelihoods.log_Normal(tbin, inner_dims, inv_link, sigma_t, 
                                                 allow_duplicate=True, tensor_type=tensor_type)
        
    elif ll_mode_comps[0] == 'U':
        basis = get_basis(ll_mode_comps[1])
        mapping_net = net(C, basis, max_count, neurons, False)
        likelihood = nprb.likelihoods.Universal(
            neurons, C, inv_link, max_count, mapping_net, tensor_type=tensor_type)
        
    else:
        raise NotImplementedError
        
    return likelihood



def gen_name(model_dict, delay, fold):
    delaystr = ''.join(str(d) for d in model_dict['delays'])
    
    name = model_dict['model_name'] + '_{}_{}H{}_{}_X[{}]_Z[{}]_{}K{}_{}d{}_{}f{}'.format(
        model_dict['ll_mode'], model_dict['filt_mode'], model_dict['hist_len'], model_dict['map_mode'], 
        model_dict['x_mode'], model_dict['z_mode'], model_dict['bin_size'], model_dict['max_count'], 
        delaystr, delay, model_dict['folds'], fold, 
    )
    return name



### script ###
def standard_parser(usage, description):
    """
    Parser arguments belonging to training loop
    """
    parser = argparse.ArgumentParser(
        usage=usage, description=description
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
    )
    
    parser.add_argument('--tensor_type', default='float', action='store', type=str)
    
    parser.add_argument('--batch_size', default=10000, type=int)
    parser.add_argument('--cv', nargs='+', type=int)
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--bin_size', type=int)
    parser.add_argument('--single_spikes', dest='single_spikes', action='store_true')
    parser.set_defaults(single_spikes=False)
    
    parser.add_argument('--ncvx', default=2, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--cov_MC', default=1, type=int)
    parser.add_argument('--ll_MC', default=10, type=int)
    parser.add_argument('--integral_mode', default='MC', action='store', type=str)
    
    parser.add_argument('--jitter', default=1e-5, type=float)
    parser.add_argument('--max_epochs', default=3000, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_2', default=1e-3, type=float)
    
    parser.add_argument('--scheduler_factor', default=0.9, type=float)
    parser.add_argument('--scheduler_interval', default=100, type=int)
    parser.add_argument('--loss_margin', default=-1e0, type=float)
    parser.add_argument('--margin_epochs', default=100, type=int)
    
    parser.add_argument('--likelihood', action='store', type=str)
    parser.add_argument('--filter', default='', action='store', type=str)
    parser.add_argument('--mapping', default='', action='store', type=str)
    parser.add_argument('--x_mode', default='', action='store', type=str)
    parser.add_argument('--z_mode', default='', action='store', type=str)
    parser.add_argument('--delays', nargs='+', default=[0], type=int)
    parser.add_argument('--hist_len', default=0, type=int)
    return parser



def preprocess_data(dataset_dict, folds, delays, cv_runs, batchsize, 
                    hist_len, has_latent=False, trial_sizes=None):
    """
    Returns delay shifted cross-validated data for training
    rcov list of arrays of shape (neurons, time, 1)
    rc_t array of shape (trials, neurons, time) or (neurons, time)
    
    Data comes in as stream of data, trials are appended consecutively
    """
    rc_t, resamples, rcov = dataset_dict['spiketrains'], dataset_dict['timesamples'], dataset_dict['covariates']
    returns = []
    
    # need trial as tensor dimension for these
    if trial_sizes is not None:
        if delays != [0]:
            raise ValueError('Delays not supported in appended trials')
        if hist_len > 0:
            raise ValueError('Coupling filter not supported in appended trials')
    
    if delays != [0]:
        if min(delays) > 0:
            raise ValueError('Delay minimum must be 0 or less')
        if max(delays) < 0:
            raise ValueError('Delay maximum must be 0 or less')
            
        D_min = -min(delays)
        D_max = -max(delays)
        dd = delays
        
    else:
        D_min = 0
        D_max = 0
        dd = [0]
        
    # history of spike train filter
    rcov = {n: rc[hist_len:] for n, rc in rcov.items()}
    resamples -= hist_len
    
    D = -D_max+D_min # total delay steps - 1
    for delay in dd:
        
        # delays
        rc_t_ = rc_t[..., D_min:(D_max if D_max < 0 else None)]
        _min = D_min+delay
        _max = D_max+delay
        
        rcov_ = {n: rc[_min:(_max if _max < 0 else None)] for n, rc in rcov.items()}
        resamples_ = resamples - D
        
        # get cv datasets
        if trial_sizes is not None and has_latent: # trials and has latent
            cv_sets, cv_inds = utils.neural.spiketrials_CV(folds, rc_t_, resamples_, rcov_, trial_sizes)
        else:
            cv_sets, vstart = utils.neural.spiketrain_CV(folds, rc_t_, resamples_, rcov_, spk_hist_len=hist_len)

        for kcv in cv_runs:
            if kcv >= 0: # CV data
                ftrain, fcov, vtrain, vcov = cv_sets[kcv]

                if has_latent:
                    if trial_sizes is None: # continual, has latent and CV, removed validation segment is temporal disconnect
                        segment_lengths = [vstart[kcv], resamples_-vstart[kcv]-vtrain.shape[-1]]
                        trial_ids = [0]*len(segment_lengths)
                        fbatch_info = utils.neural.batch_segments(segment_lengths, trial_ids, batchsize)
                        vbatch_info = batchsize

                    else:
                        ftr_inds, vtr_inds = cv_inds[kcv]
                        fbatch_info = utils.neural.batch_segments([trial_sizes[ind] for ind in ftr_inds], ftr_inds, batchsize)
                        vbatch_info = utils.neural.batch_segments([trial_sizes[ind] for ind in vtr_inds], vtr_inds, batchsize)

                else:
                    fbatch_info = batchsize
                    vbatch_info = batchsize

            else: # full data
                ftrain, fcov = rc_t_, rcov_
                if trial_sizes is not None and has_latent:
                    trial_ids = list(range(len(trial_sizes)))
                    fbatch_info = utils.neural.batch_segments(trial_sizes, trial_ids, batchsize)
                else:
                    fbatch_info = batchsize
                        
                vtrain, vcov = None, None
                vbatch_info = None

            preprocess_dict = {
                'fold': kcv, 
                'delay': delay,
                'spiketrain_fit': ftrain, 
                'covariates_fit': fcov, 
                'batching_info_fit': fbatch_info, 
                'spiketrain_val': vtrain, 
                'covariates_val': vcov, 
                'batching_info_val': vbatch_info, 
            }
            returns.append(preprocess_dict)
        
    return returns
    

    
def setup_model(data_tuple, model_dict, enc_used):
    """"
    Assemble the encoding model
    """
    spktrain, cov, batch_info = data_tuple
    neurons, timesamples = spktrain.shape[0], spktrain.shape[-1]
    
    ll_mode, filt_mode, map_mode, x_mode, z_mode, tensor_type = \
        model_dict['ll_mode'], model_dict['filt_mode'], model_dict['map_mode'], \
        model_dict['x_mode'], model_dict['z_mode'], model_dict['tensor_type']
    
    # seed everything
    seed = model_dict['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # checks
    if filt_mode == '' and model_dict['hist_len'] > 0:
        raise ValueError
    
    # inputs
    input_data, d_x, d_z = inputs_used(model_dict, cov, batch_info)
    model_dict['map_xdims'], model_dict['map_zdims'] = d_x, d_z
    
    input_group = nprb.inference.input_group(tensor_type)
    input_group.set_XZ(input_data, timesamples, batch_info=batch_info)
    
    # encoder mapping
    ll_mode_comps = ll_mode.split('-')
    C = int(ll_mode_comps[2]) if ll_mode_comps[0] == 'U' else 1
    model_dict['map_outdims'] = neurons*C # number of output dimensions of the input_mapping
    
    learn_mean = (ll_mode_comps[0] != 'U')
    mapping = enc_used(model_dict, cov, learn_mean)

    # likelihood
    likelihood = get_likelihood(model_dict, enc_used)
    if filt_mode != '':
        filterobj = coupling_filter(model_dict)
        likelihood = nprb.likelihoods.filters.filtered_likelihood(likelihood, filterobj)
    likelihood.set_Y(torch.from_numpy(spktrain), batch_info=batch_info)
    
    full = nprb.inference.VI_optimized(input_group, mapping, likelihood)
    return full

    

def train_model(dev, parser, dataset_dict, enc_used, checkpoint_dir, trial_sizes=None):
    """
    General training loop
    
    def inputs_used(model_dict, covariates, batch_info):
        Get inputs for model

    def enc_used(model_dict, covariates, inner_dims):
        Function for generating encoding model
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
            
    nonconvex_trials = parser.ncvx
    seed = parser.seed
    
    if parser.tensor_type == 'float':
        tensor_type = torch.float
    elif parser.tensor_type == 'double':
        tensor_type = torch.double
    else:
        raise ValueError('Invalid tensor type in arguments')

    folds = parser.cv_folds
    delays = parser.delays
    cv_runs = parser.cv
    batchsize = parser.batch_size
    integral_mode = parser.integral_mode
    
    # mode
    ll_mode = parser.likelihood
    filt_mode = parser.filter
    map_mode = parser.mapping
    x_mode = parser.x_mode
    z_mode = parser.z_mode
    hist_len = parser.hist_len
    
    model_dict = {
        'seed': seed, 
        'll_mode': ll_mode, 
        'filt_mode': filt_mode, 
        'map_mode': map_mode, 
        'x_mode': x_mode, 
        'z_mode': z_mode, 
        'hist_len': hist_len, 
        'folds': folds, 
        'delays': delays, 
        'neurons': dataset_dict['neurons'], 
        'max_count': dataset_dict['max_count'], 
        'bin_size': dataset_dict['bin_size'], 
        'tbin': dataset_dict['tbin'], 
        'model_name': dataset_dict['name'], 
        'tensor_type': tensor_type, 
        'jitter': parser.jitter, 
    }
    

    # training
    has_latent = False if z_mode == '' else True
    preprocessed = preprocess_data(
        dataset_dict, folds, delays, cv_runs, batchsize, hist_len, has_latent, trial_sizes
    )
    for cvdata in preprocessed:
        fitdata = (
            cvdata['spiketrain_fit'], 
            cvdata['covariates_fit'], 
            cvdata['batching_info_fit'], 
        )
        model_name = gen_name(model_dict, cvdata['delay'], cvdata['fold'])
        print(model_name)
            
        ### fitting ###
        for kk in range(nonconvex_trials):

            retries = 0
            while True:
                try:
                    # model
                    full_model = setup_model(
                        fitdata, model_dict, enc_used
                    )
                    full_model.to(dev)

                    # fit
                    sch = lambda o: optim.lr_scheduler.MultiplicativeLR(o, lambda e: parser.scheduler_factor)
                    opt_tuple = (optim.Adam, parser.scheduler_interval, sch)
                    opt_lr_dict = {'default': parser.lr}
                    if z_mode == 'T1':
                        opt_lr_dict['mapping.kernel.kern1._lengthscale'] = parser.lr_2
                    for z_dim in full_model.input_group.latent_dims:
                        opt_lr_dict['input_group.input_{}.variational.finv_std'.format(z_dim)] = parser.lr_2

                    full_model.set_optimizers(opt_tuple, opt_lr_dict)#, nat_grad=('rate_model.0.u_loc', 'rate_model.0.u_scale_tril'))

                    annealing = lambda x: 1.0
                    losses = full_model.fit(parser.max_epochs, loss_margin=parser.loss_margin, 
                                            margin_epochs=parser.margin_epochs, kl_anneal_func=annealing, 
                                            cov_samples=parser.cov_MC, ll_samples=parser.ll_MC, ll_mode=integral_mode)
                    break
                    
                except RuntimeError as e:
                    print(e)
                    print('Retrying...')
                    if retries == 3: # max retries
                        print('Stopped after max retries.')
                        sys.exit()
                    retries += 1

            ### save and progress ###
            if os.path.exists(checkpoint_dir + model_name + '_result.p'):  # check previous best losses
                with open(checkpoint_dir + model_name + '_result.p', 'rb') as f:
                    results = pickle.load(f)
                    lowest_loss = results['training_loss'][-1]
            else:        
                lowest_loss = np.inf # nonconvex pick the best

            if losses[-1] < lowest_loss:
                # save model
                torch.save({'full_model': full_model.state_dict()}, checkpoint_dir + model_name + '.pt')
                
                with open(checkpoint_dir + model_name + '_result.p', 'wb') as f:
                    results = {'training_loss': losses}
                    pickle.dump(results, f)


                
                
def load_model(checkpoint_dir, model_dict, dataset_dict, enc_used, 
               delay, cv_run, batch_info, gpu, trial_sizes=None):
    """
    Load the model with cross-validated data structure
    """
    ### data ###
    has_latent = False if model_dict['z_mode'] == '' else True
    cvdata = preprocess_data(dataset_dict, model_dict['folds'], [delay], [cv_run], 
                             batch_info, model_dict['hist_len'], has_latent, trial_sizes)[0]
    
    fit_data = (
        cvdata['spiketrain_fit'], 
        cvdata['covariates_fit'], 
        cvdata['batching_info_fit'], 
    )
    val_data = (
        cvdata['spiketrain_val'], 
        cvdata['covariates_val'], 
        cvdata['batching_info_val'], 
    )
    fit_set = (
        inputs_used(model_dict, fit_data[1], batch_info)[0], 
        torch.from_numpy(fit_data[0]), 
        fit_data[2],
    )
    validation_set = (
        inputs_used(model_dict, val_data[1], batch_info)[0] if val_data[1] is not None else None, 
        torch.from_numpy(val_data[0]) if val_data[0] is not None else None, 
        val_data[2],
    )
    
    ### model ###
    full_model = setup_model(fit_data, model_dict, enc_used)
    full_model.to(gpu)
    
    ### load ###
    model_name = gen_name(model_dict, delay, cv_run)
    checkpoint = torch.load(checkpoint_dir + model_name + '.pt', map_location='cuda:{}'.format(gpu))
    full_model.load_state_dict(checkpoint['full_model'])
    with open(checkpoint_dir + model_name + '_result.p', 'rb') as f:
        training_results = pickle.load(f)
        
    return full_model, training_results, fit_set, validation_set