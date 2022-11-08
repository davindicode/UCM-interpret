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

# get GPU device if available
gpu_dev = 0
dev = utils.pytorch.get_device(gpu=gpu_dev)

import warnings
warnings.simplefilter('ignore')

#mice_sessions = {
#    'Mouse12': ['120806', '120807', '120809', '120810' ], # '120808' is missing position files
#    'Mouse17': ['130125', '130128', '130129', '130130', '130131', '130201', '130202', '130203', '130204'],
#    'Mouse20': ['130514', '130515', '130516', '130517', '130520'],
#    'Mouse24': ['131213', '131216', '131217','131218'],
#    'Mouse25': ['140123', '140124', '140128', '140129', '140130', '140131', '140203', '140204', '140205', '140206'],
#    'Mouse28': ['140310', '140311', '140312', '140313', '140317', '140318'],
#} 

mice_sessions = {
    'Mouse24': ['131213'],
    'Mouse25': ['140129']
} 

data_dir = '/scratches/ramanujan_2/dl543/HDC_PartIII/'
models_dir = '/scratches/ramanujan_2/dl543/HDC_PartIII/checkpoint/'
savedir = '/scratches/ramanujan_2/vn283/HDC_PartIII/tc_data/'

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


def tuning_curve_1d(cov_name, use_neuron, modelfit, rcov, tbin, num_points=100,
                    MC=30, batch_size=1000):
    """Compute marginalized tuning curve for a given covariate."""
    lower_limit = np.min(rcov[cov_name])
    upper_limit = np.max(rcov[cov_name])
    sweep = torch.linspace(lower_limit, upper_limit, num_points)[None, :]

    rcov_matrix = [torch.tensor(rcov[k]) for k in rcov.keys()]
    with torch.no_grad():
        P_mc = lib.helper.marginalized_P(modelfit, sweep,
                                         [list(rcov.keys()).index(cov_name)],
                                         rcov_matrix, batch_size, use_neuron,
                                         MC)

    K = P_mc.shape[-1]
    counts = torch.arange(K)

    hd_mean = (counts[None, None, None, :] * P_mc).sum(
        -1)  # (MC, neurons, steps)
    hd_rate = hd_mean / tbin  # in units of Hz
    hd_var = (counts[None, None, None, :] ** 2 * P_mc).sum(-1) - hd_mean ** 2
    hd_FF = hd_var / (hd_mean + 1e-12)
    return hd_rate, hd_FF, sweep


def tuning_index(hd_stat):
    """Compute the tuning index of a tuning curve with a given statistics of spike count distributions."""
    _, tc_mean, _ = utils.signal.percentiles_from_samples(hd_stat, [0.05, 0.5, 0.95])

    tc_max, _ = torch.max(tc_mean, axis=1)
    tc_min, _ = torch.min(tc_mean, axis=1)

    return (tc_max - tc_min) / (tc_max + tc_min)


def tuning_index_features(dataset, modelfit, model_dict):
    """Compute feature vector consisting of tuning indices for different covariates."""
    features_rate = []
    features_ff = []
    for cov in ['hd', 'omega', 'speed', 'x', 'y', 'time']:
        print('Calculating tuning indices for ', cov, '\n')
        hd_rate, hd_FF, sweep = tuning_curve_1d(cov, list(range(dataset['neurons'])),
                                                modelfit,
                                                dataset['covariates'],
                                                model_dict['tbin'],
                                                batch_size=100)

        features_rate.append(tuning_index(hd_rate))
        features_ff.append(tuning_index(hd_FF))

    f_rate = np.array([features_rate[i].numpy() for i in range(len(features_rate))])  # (num_cov, num_neurons)
    features_rate = np.swapaxes(f_rate, 0, 1)  # num_neurons x num_covariates

    f_ff = np.array(
        [features_ff[i].numpy() for i in range(len(features_ff))])
    features_ff = np.swapaxes(f_ff, 0, 1)  # num_neurons x num_covariates

    return features_rate, features_ff


def tuning_curves(dataset, modelfit, model_dict, num_steps=100, MC=30, batch_size=100):
    """Compute marginalized tuning curves for all covariates."""

    features_rate = np.empty((dataset['neurons'], num_steps))
    features_ff = np.empty((dataset['neurons'], num_steps))
    covariates = np.empty((num_steps,))
    for cov in ['hd', 'omega', 'speed', 'x', 'y', 'time']:
        print('\n Calculating tuning indices for ', cov)
        hd_rate, hd_FF, sweep = tuning_curve_1d(cov, list(range(dataset['neurons'])),
                                                modelfit,
                                                dataset['covariates'],
                                                model_dict['tbin'],
                                                num_points=num_steps,
                                                batch_size=batch_size,
                                                MC=MC)

        _, hd_rate_mean, _ = utils.signal.percentiles_from_samples(hd_rate,
                                                              [0.05, 0.5,
                                                               0.95])
        _, hd_ff_mean, _ = utils.signal.percentiles_from_samples(hd_FF,
                                                              [0.05, 0.5,
                                                               0.95]) # (neurons, steps)
        np.append(features_rate, hd_rate_mean.numpy(), axis=0)  # (num_covariates, neurons, steps)
        np.append(features_ff, hd_ff_mean.numpy(), axis=0)
        np.append(covariates, sweep.numpy().flatten(), axis=0)  # (num_covariates, steps)

    features_rate = np.swapaxes(features_rate, 0, 1)
    features_ff = np.swapaxes(features_ff, 0, 1)

    return features_rate, features_ff, covariates


def compute_and_save_tcs(mouse_id, session_id):
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
    features_rate_nonhdc, features_ff_nonhdc, covariates_nonhdc = tuning_curves(dataset_nonhdc, modelfit_nonhdc, model_dict_nonhdc)
    print('HD neurons dataset')
    features_rate_hdc, features_ff_hdc, covariates_hdc = tuning_curves(dataset_hdc, modelfit_hdc, model_dict_hdc)

    # Save data
    print('saving data')
    np.savez_compressed(savedir + f'{mouse_id}_{session_id}_{phase}_hdc',
                        tuning_curves_rates=features_rate_hdc,
                        tuning_curves_FF=features_ff_hdc,
                        tuning_curves_covariates=covariates_hdc)

    np.savez_compressed(savedir + f'{mouse_id}_{session_id}_{phase}_nonhdc',
                        tuning_curves_rates=features_rate_nonhdc,
                        tuning_curves_FF=features_ff_nonhdc,
                        tuning_curves_covariates=covariates_nonhdc)


def main():
    for mouse_id in mice_sessions.keys():
        for session_id in mice_sessions[mouse_id]:
            compute_and_save_tcs(mouse_id, session_id)


if __name__ == "__main__":
    main()
