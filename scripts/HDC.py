import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


import pickle

import sys
sys.path.append("..")


import neuroprob as nprb
from neuroprob import utils

import lib




### data ###
def get_dataset(mouse_id, session_id, phase, subset, bin_size, single_spikes, path):

    data = np.load(path + '{}_{}_{}.npz'.format(mouse_id, session_id, phase))
    spktrain = data['spktrain']
    hdc_unit = data['hdc_unit']
    x_t = data['x_t']
    y_t = data['y_t']
    hd_t = data['hd_t']
    neuron_regions = data['neuron_regions']
    print('units: ', spktrain.shape[0], ' hdc units: ', hdc_unit.sum())
    
    sample_bin = 0.001

    neurons = spktrain.shape[0]
    track_samples = spktrain.shape[1]

    tbin, resamples, rc_t, (rhd_t, rx_t, ry_t) = utils.neural.bin_data(
        bin_size, sample_bin, spktrain, track_samples, 
        (np.unwrap(hd_t), x_t, y_t), average_behav=True, binned=True
    )

    # recompute velocities
    rw_t = (rhd_t[1:]-rhd_t[:-1])/tbin
    rw_t = np.concatenate((rw_t, rw_t[-1:]))

    rvx_t = (rx_t[1:]-rx_t[:-1])/tbin
    rvy_t = (ry_t[1:]-ry_t[:-1])/tbin
    rs_t = np.sqrt(rvx_t**2 + rvy_t**2)
    rs_t = np.concatenate((rs_t, rs_t[-1:]))
    rtime_t = np.arange(resamples)*tbin

    rcov = {
        'hd': rhd_t % (2*np.pi), 
        'omega': rw_t, 
        'speed': rs_t, 
        'x': rx_t, 
        'y': ry_t, 
        'time': rtime_t,
    }
    
    if single_spikes is True:
        rc_t[rc_t > 1.] = 1.
        
    if subset == 'hdc':
        rc_t = rc_t[hdc_unit, :]
    elif subset == 'nonhdc':
        rc_t = rc_t[~hdc_unit, :]
    elif subset != 'all':
        raise ValueError('Invalid data subset')
        
    units_used = rc_t.shape[0]
    max_count = int(rc_t.max())
    
    name = 'th1-{}-{}-{}-{}'.format(mouse_id, session_id, phase, subset)
    metainfo = {
        'neuron_regions': neuron_regions,
    }
    
    dataset_dict = {
        'name': name,
        'covariates': rcov, 
        'spiketrains': rc_t, 
        'neurons': units_used, 
        'metainfo': metainfo,
        'tbin': tbin,
        'timesamples': resamples,
        'max_count': max_count,
        'bin_size': bin_size, 
    }
    return dataset_dict



### model ###
def enc_used(model_dict, covariates, learn_mean):
    """
    """
    ll_mode, map_mode, x_mode, z_mode = model_dict['ll_mode'], model_dict['map_mode'], \
        model_dict['x_mode'], model_dict['z_mode']
    jitter, tensor_type = model_dict['jitter'], model_dict['tensor_type']
    neurons, in_dims = model_dict['neurons'], model_dict['map_xdims'] + model_dict['map_zdims']
    
    out_dims = model_dict['map_outdims']
    mean = torch.zeros((out_dims)) if learn_mean else 0  # not learnable vs learnable
    
    map_mode_comps = map_mode.split('-')
    x_mode_comps = x_mode.split('-')
    
    def get_inducing_locs_and_ls(comp):
        if comp == 'hd':
            locs = np.linspace(0, 2*np.pi, num_induc+1)[:-1]
            ls = 5.*np.ones(out_dims)
        elif comp == 'omega':
            scale = covariates['omega'].std()
            locs = scale * np.random.randn(num_induc)
            ls = scale * np.ones(out_dims)
        elif comp == 'speed':
            scale = covariates['speed'].std()
            locs = np.random.uniform(0, scale, size=(num_induc,))
            ls = 10.*np.ones(out_dims)
        elif comp == 'x':
            left_x = covariates['x'].min()
            right_x = covariates['x'].max()
            locs = np.random.uniform(left_x, right_x, size=(num_induc,))
            ls = (right_x - left_x)/10. * np.ones(out_dims)
        elif comp == 'y':
            bottom_y = covariates['y'].min()
            top_y = covariates['y'].max()
            locs = np.random.uniform(bottom_y, top_y, size=(num_induc,))
            ls = (top_y - bottom_y)/10. * np.ones(out_dims)
        elif comp == 'time':
            scale = covariates['time'].max()
            locs = np.linspace(0, scale, num_induc)
            ls = scale/2. * np.ones(out_dims)
        else:
            raise ValueError('Invalid covariate type')
            
        return locs, ls, (comp == 'hd')
    
    
    if map_mode_comps[0] == 'svgp':
        num_induc = int(map_mode_comps[1])

        var = 1.0 # initial kernel variance
        v = var*torch.ones(out_dims)

        ind_list = []
        kernel_tuples = [('variance', v)]
        ang_ls, euclid_ls = [], []
        
        # x
        for xc in x_mode_comps:
            if xc == '':
                continue
                
            locs, ls, angular = get_inducing_locs_and_ls(xc)
            
            ind_list += [locs]
            if angular:
                ang_ls += [ls]
            else:
                euclid_ls += [ls]
            
        if len(ang_ls) > 0:
            kernel_tuples += [('SE', 'torus', torch.tensor(ang_ls))]
        if len(euclid_ls) > 0:
            kernel_tuples += [('SE', 'euclid', torch.tensor(euclid_ls))]

        # z
        latent_k, latent_u = lib.models.latent_kernel(z_mode, num_induc, out_dims)
        kernel_tuples += latent_k
        ind_list += latent_u

        # objects
        kernelobj, constraints = lib.models.create_kernel(kernel_tuples, 'exp', tensor_type)

        Xu = torch.tensor(ind_list).T[None, ...].repeat(out_dims, 1, 1)
        inpd = Xu.shape[-1]
        inducing_points = nprb.kernels.kernel.inducing_points(out_dims, Xu, constraints, 
                                                              tensor_type=tensor_type)

        mapping = nprb.mappings.SVGP(
            in_dims, out_dims, kernelobj, inducing_points=inducing_points, 
            jitter=jitter, whiten=True, 
            mean=mean, learn_mean=learn_mean, 
            tensor_type=tensor_type
        )
        
    else:
        raise ValueError
        
    return mapping



### main ###
def main():
    parser = lib.models.standard_parser("%(prog)s [OPTION] [FILE]...", "Fit model to data.")
    parser.add_argument('--data_path', action='store', type=str)
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', action='store', type=str)
    
    parser.add_argument('--mouse_id', action='store', type=str)
    parser.add_argument('--session_id', action='store', type=str)
    parser.add_argument('--phase', action='store', type=str)  # 'sleep', 'wake'
    parser.add_argument('--subset', action='store', type=str)  # 'hdc', 'nonhdc', 'all'
    args = parser.parse_args()
    
    dev = utils.pytorch.get_device(gpu=args.gpu)
    
    mouse_id, session_id, phase, subset = args.mouse_id, args.session_id, args.phase, args.subset
    dataset_dict = get_dataset(mouse_id, session_id, phase, subset, 
                               args.bin_size, args.single_spikes, args.data_path)

    lib.models.train_model(dev, args, dataset_dict, enc_used, args.checkpoint_dir)

                


if __name__ == "__main__":
    main()