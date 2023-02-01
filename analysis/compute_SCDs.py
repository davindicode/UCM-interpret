import numpy as np
import torch

# add paths to access shared code
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts/"))

# import library implementing models
from neuroprob import utils

# import utility code for model building/training/loading
import lib
import HDC

import warnings
warnings.simplefilter('ignore')

# mice_sessions = {
#    'Mouse12': ['120806', '120807', '120809', '120810' ], # '120808' is missing position files
#    'Mouse17': ['130125', '130128', '130129', '130131', '130201', '130202', '130203', '130204'], # '130130' broken
#    'Mouse20': ['130514', '130515', '130516', '130517'], # '130520' hdc models fails
#    'Mouse24': ['131213', '131216', '131217','131218'],
#    'Mouse25': ['140123', '140124', '140128', '140129', '140130', '140131', '140203', '140204', '140205', '140206'],
#    'Mouse28': ['140310', '140311', '140312', '140313', '140317', '140318'],
#} 

mice_sessions = {
    'Mouse12': ['120806', '120807'],
    'Mouse17': ['130125', '130128', '130131', '130202', '130203'],
    'Mouse20': ['130514', '130515', '130516', '130517'],
    'Mouse24': ['131213', '131217', '131218'],
    'Mouse25': ['140124', '140128', '140129'],
    'Mouse28': ['140310']
} 

mice_sessions = {'Mouse12': ['120806', '120807']}
# already done
#    'Mouse12': [],
#    'Mouse17': [],
#    'Mouse20': [],
#    'Mouse24': [],
#    'Mouse25': [],
#    'Mouse28': []


data_dir = '/scratches/ramanujan_2/dl543/HDC_PartIII/'
models_dir = '/scratches/ramanujan_2/vn283/HDC_PartIII/checkpoint/'
savedir = '/scratches/ramanujan_2/vn283/HDC_PartIII/scd_data/'

phase = 'wake'
bin_size = 160  # ms
single_spikes = False

cv_run = -1  # test set is last 1/5 of dataset time series
delay = 0
batch_size = 5000  # size of time segments of each batch in dataset below


def get_model_dict(dataset_dict):
    """Get model info needed for loading."""
    return {
        'seed': 123,
        'll_mode': 'U-ec-3',  # stands for universal count model with exponential-quadratic expansion and C = 3
        'filt_mode': '',  # GLM couplings
        'map_mode': 'svgp-64',  # a sparse variational GP mapping with 64 inducing points
        'x_mode': 'hd-omega-speed-x-y-time',  # observed covariates (behaviour)
        'z_mode': '',  # latent covariates
        'hist_len': 0,
        'folds': 5,
        'delays': [0],
        'neurons': dataset_dict['neurons'],
        'max_count': dataset_dict['max_count'],
        'bin_size': dataset_dict['bin_size'],
        'tbin': dataset_dict['tbin'],
        'model_name': dataset_dict['name'],
        'tensor_type': torch.float,
        'jitter': 1e-5,
    }


def tuning_curve(cov_name, use_neuron, modelfit, rcov, tbin, num_points=100,
                    MC=30, batch_size=1000, skip=1):
    """Compute marginalized tuning curve for a given covariate."""
    lower_limit = np.min(rcov[cov_name])
    upper_limit = np.max(rcov[cov_name])
    sweep = torch.linspace(lower_limit, upper_limit, num_points)[None, :]

    rcov_matrix = [torch.tensor(rcov[k]) for k in rcov.keys()]
    with torch.no_grad():
        P_mc = lib.helper.marginalized_P(modelfit, sweep,
                                         [list(rcov.keys()).index(cov_name)],
                                         rcov_matrix, batch_size, use_neuron,
                                         MC, skip=skip).mean(0).cpu().numpy() # (neurons, steps, count)
    return P_mc, sweep.cpu().numpy().flatten()


def tuning_curves(dataset, modelfit, model_dict, num_steps=100, MC=30, batch_size=100, skip=1):
    """Compute marginalized tuning curves for all covariates."""

    features_rate = np.empty((0, dataset['neurons'], num_steps))
    features_ff = np.empty((0, dataset['neurons'], num_steps))
    covariates = np.empty((0, num_steps))
    for cov in ['hd']:
        print('\n Calculating tuning indices for ', cov)
        P_mc, sweep = tuning_curve_1d(cov, list(range(dataset['neurons'])),
                                                modelfit,
                                                dataset['covariates'],
                                                model_dict['tbin'],
                                                num_points=num_steps,
                                                batch_size=batch_size,
                                                MC=MC, skip=skip)  # (neurons, steps, count)
        features_rate = np.concatenate((features_rate, hd_rate_mean.numpy()[None, :]), axis=0)  # (num_covariates, neurons, steps)
        covariates = np.concatenate((covariates, sweep[None, :]), axis=0)  # (num_covariates, steps)

    features_rate = np.swapaxes(features_rate, 0, 1)
    features_ff = np.swapaxes(features_ff, 0, 1)

    return features_rate, features_ff, covariates


def compute_and_save_tcs(mouse_id, session_id, gpu_dev=0):
    """Compute and save tuning curves for a given mouse and session"""
    # Loading data
    dataset_hdc = HDC.get_dataset(mouse_id, session_id, phase, 'hdc', bin_size,
                                  single_spikes, path=data_dir)

    dataset_nonhdc = HDC.get_dataset(mouse_id, session_id, phase, 'nonhdc',
                                     bin_size, single_spikes, path=data_dir)

    # Loading the models
    model_dict_hdc = get_model_dict(dataset_hdc)
    model_dict_nonhdc = get_model_dict(dataset_nonhdc)

    modelfit_hdc, training_results_hdc, fit_set_hdc, validation_set_hdc = lib.models.load_model(
        models_dir, model_dict_hdc, dataset_hdc, HDC.enc_used,
        delay, cv_run, batch_size, gpu_dev
    )

    modelfit_nonhdc, training_results_nonhdc, fit_set_nonhdc, validation_set_nonhdc = lib.models.load_model(
        models_dir, model_dict_nonhdc, dataset_nonhdc, HDC.enc_used,
        delay, cv_run, batch_size, gpu_dev
    )

    # Compute tuning curves
    print('Non-hd neurons dataset')
    P_nonhdc, sweep_nonhdc = tuning_curve('hd', list(range(dataset_nonhdc['neurons'])),
                                                modelfit_nonhdc,
                                                dataset_nonhdc['covariates'],
                                                model_dict_nonhdc['tbin'],
                                                skip=3)
    print('HD neurons dataset')
    P_hdc, sweep_hdc = tuning_curve('hd', list(range(dataset_hdc['neurons'])),
                                                modelfit_hdc,
                                                dataset_hdc['covariates'],
                                                model_dict_hdc['tbin'],
                                                skip=3)

    # Save data
    print('saving data')
    np.savez_compressed(savedir + f'{mouse_id}_{session_id}_{phase}_hdc',
                        scd=P_hdc, covariates=sweep_hdc)

    np.savez_compressed(savedir + f'{mouse_id}_{session_id}_{phase}_nonhdc',
                        scd=P_nonhdc, covariates=sweep_nonhdc)


def main():
    parser = lib.models.standard_parser("%(prog)s [OPTION] [FILE]...", "Compute tuning curves.")
    parser.add_argument('--mouse_id', action='store', type=str)
    parser.add_argument('--session_id', action='store', type=str)
    args = parser.parse_args()
    
    dev = utils.pytorch.get_device(gpu=args.gpu)
    
    mouse_id, session_id = args.mouse_id, args.session_id

    print(f'{mouse_id}-{session_id}')
    compute_and_save_tcs(mouse_id, session_id, args.gpu)


if __name__ == "__main__":
    main()
