import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import numbers

from .. import base




### Mixtures ###
class _mappings(base._input_mapping):
    """
    """
    def __init__(self, input_dims, mappings, inv_link):
        """
        Additive fields, so exponential inverse link function.
        All models should have same the input and output structure.
        
        :params list models: list of base models, each initialized separately
        """
        self.maps = len(mappings)
        if self.maps < 2:
            raise ValueError('Need to have more than one component mapping')
        
        self.inv_link = inv_link

        #covar_type = None # intially no uncertainty in model
        for m in range(len(mappings)): # consistency check
            if m < len(mappings)-1: # consistency check
                if mappings[m].out_dims != mappings[m+1].out_dims:
                    raise ValueError('Mappings do not match in output dimensions')
                if mappings[m].tensor_type != mappings[m+1].tensor_type:
                    raise ValueError('Tensor types of mappings do not match')
        
        super().__init__(input_dims, mappings[0].out_dims, mappings[0].tensor_type)
            
        self.mappings = nn.ModuleList(mappings)
        
        
    def constrain(self):
        for m in self.mappings:
            m.constrain()
        
        
    def KL_prior(self):
        KL_prior = 0
        for m in self.mappings:
            KL_prior = KL_prior + m.KL_prior()
        return KL_prior



class mixture_model(_mappings):
    """
    Takes in identical base models to form a mixture model.
    """
    def __init__(self, input_dims, mappings, inv_link='relu'):
        super().__init__(input_dims, mappings, inv_link)
    
    
    def compute_F(self, XZ):
        """
        Note that the rates used for addition in the mixture model are evaluated as the 
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the 
        posterior mean. The difference between these quantities is small when the posterior 
        variance is small.
        
        """
        var = 0
        r_ = 0
        for m in self.mappings:
            F_mu, F_var = m.compute_F(XZ)
            if isinstance(F_var, numbers.Number) is False:
                var = var + (base._inv_link_deriv[self.inv_link](F_mu)**2*F_var) # delta method
            r_ = r_ + m.f(F_mu)
            
        return r_, var



class product_model(_mappings):
    """
    Takes in identical base models to form a product model.
    """
    def __init__(self, input_dims, mappings, inv_link='relu'):
        super().__init__(input_dims, mappings, inv_link)
    
    
    def compute_F(self, XZ):
        """
        Note that the rates used for multiplication in the product model are evaluated as the 
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the 
        posterior mean. The difference between these quantities is small when the posterior 
        variance is small.
        
        The exact method would be to use MC sampling.
        
        :param torch.Tensor cov: input covariates of shape (sample, time, dim)
        """
        rate_ = []
        var_ = []
        for m in self.mappings:
            F_mu, F_var = m.compute_F(XZ)
            rate_.append(m.f(F_mu))
            if isinstance(F_var, numbers.Number):
                var_.append(0)
                continue
            var_.append(base._inv_link_deriv[self.inv_link](F_mu)**2*F_var) # delta method
                
        tot_var = 0
        rate_ = torch.stack(rate_, dim=0)
        for m, var in enumerate(var_):
            ind = torch.tensor([i for i in range(rate_.shape[0]) if i != m])
            if isinstance(var, numbers.Number) is False:
                tot_var = tot_var + (rate_[ind]).prod(dim=0)**2*var
            
        return rate_.prod(dim=0), tot_var

    
    
class mixture_composition(_mappings):
    """
    Takes in identical base models to form a mixture model with custom functions.
    Does not support models with variational uncertainties, ignores those.
    """
    def __init__(self, input_dims, mappings, comp_func, inv_link='relu'):
        super().__init__(input_dims, mappings, inv_link)
        self.comp_func = comp_func
    
    
    def compute_F(self, XZ):
        """
        Note that the rates used for addition in the mixture model are evaluated as the 
        quantities :math:`f(\mathbb{E}[a])`, different to :math:`\mathbb{E}[f(a)]` that is the 
        posterior mean. The difference between these quantities is small when the posterior 
        variance is small.
        
        """
        r_ = [base._inv_link[self.inv_link](m.compute_F(XZ)[0]) for m in self.mappings]
        return self.comp_func(r_), 0
    
    
    
### Discrete dynamics ###
class _dynamic_mixture(_mappings):
    """
    """
    def __init__(self, input_dims, mappings, inv_link):
        super().__init__(input_dims, mappings, inv_link)
        
        
    def slice_F(self, F, discrete):
        """
        """
        TT = F.shape[0]
        d_ = discrete[:, None, :].expand(TT, self.likelihood.F_dims, timesteps)
        z_ = torch.arange(TT)[:, None, None].expand(TT, self.F_dims, timesteps).to(d_.device)
        n_ = torch.arange(self.F_dims)[None, :, None].expand(TT, self.F_dims, timesteps).to(d_.device)
        t_ = torch.arange(timesteps)[None, None, :].expand(TT, self.F_dims, timesteps).to(d_.device)
        F = F[(d_, z_, n_, t_)]
        
        return F
    


class HMM_model(_dynamic_mixture):
    """
    Finite discrete state, each state is a separate mapping model. State dynamics are provided by 
    the input, either observed or latent via some state space model.
    
    Does not support stacking HMM models.
    """
    def __init__(self, input_dims, mappings, inv_link='relu'):
        super().__init__(input_dims, mappings, inv_link)
        
        hmm_in = covariates[-1]
        covariates = covariates[:-1]
        if hmm_in is not None: # optional, could use learned values
            self.init_hmm(hmm_in, timesteps)
            
    
    def constrain(self):
        super().constrain()
        if self.maps > 1: # simplex constraint
            norm = self.hmm_T.data.sum(0)
            self.hmm_T.data /= norm[None, :]
            
            
    def KL_prior(self):
        KL_prior = super().KL_prior()
        return KL_prior
    
    
    def compute_F(self, XZ):
        """
        Compute models individually.
        
        :param torch.Tensor cov: input covariates of shape (sample, time, dim)
        """
        """maps = len(F_mu) if isinstance(F_mu, list) else 1
        if maps > 1:
            if self.mapping.state_obs:
                discrete = self.hmm_state[0].to(self.mapping.dummy.device)
            else:
                nll = torch.stack(self._nll(F_mu, F_var, maps, 'ELBO', ll_samples, 
                                            ll_mode, batch, obs_neuron, XZ, 1.)) # (state, time)
                discrete = self.mapping.sample_state(timesteps, self.logp_0, trials=self.input_group.trials, 
                                                     cond_nll=nll, viterbi=True)

            F_mu = self.slice_F(torch.stack(F_mu), discrete)
            F_var = self.slice_F(torch.stack(F_var), discrete)

        else: # only one state"""
        F_mu = [] # container for pre-nonlinearity rate value, computed over all self.F_dims (neurons)
        F_var = []
        
        for m in self.mappings: # TODO: parallel enumeration
            F_mu_, F_var_ = m.compute_F(XZ) # samples, neurons, timesteps
            F_mu.append(F_mu_)
            F_var.append(F_var_)
            
        return F_mu, F_var
    
    
    def set_hmm(self, hmm_in, batch_size=None):
        """
        Initialize the discrete latent variable state. This allows placing a HMM prior or conditioning on 
        observed discrete states.
        
        :param np.ndarray/list hmm_tuple: list contaning [hmm_T, logp_0] for latent discrete state inference or (hmm_state,) for 
                                observed latent states
        :param int/tuple batch_size: the batch size(s) in the temporal dimension
        """
        if isinstance(hmm_in, list):
            self.register_parameter('hmm_T', Parameter(torch.tensor(hmm_in[0], device=self.dummy.device, 
                                                                    dtype=self.likelihood.tensor_type)))
            self.register_parameter('logp_0', Parameter(torch.tensor(hmm_in[1], device=self.dummy.device, 
                                                                    dtype=self.likelihood.tensor_type)))
            self.hmm_obs = False
        elif isinstance(hmm_in, np.ndarray): # observed hidden state
            z = torch.tensor(hmm_in).long()
            if type(batch_size) == list:
                cb = 0
                self.hmm_state = []
                for bs in batch_size:
                    self.hmm_state.append(z[cb:cb+bs])
                    cb += bs
            else: # number
                self.hmm_state = torch.split(z, batch_size)
            self.hmm_obs = True
        else:
            raise ValueError
    
    
    def forward_hmm(self, nll, logp_0):
        """
        Discrete variable, forward algorithm or message passing to marginalize efficiently.
        
        :param torch.Tensor nll: the objective function for each latent state of shape (state, time)
        :param torch.Tensor logp_0: the initial log probabilities of the states at start of the chain
        :returns: negative log probability after marginalizing hidden states and log p(z_t|X})
        :rtype: tuple of torch.tensor
        """
        trans_logp = torch.log(self.hmm_T + 1e-12)
        for t in range(nll.shape[1]):
            if t == 0:
                logp = logp_0 - nll[:, 0] # p(z_0) * p(x_0|z_0)
            else:
                logp = (logp.unsqueeze(0) - nll[:, t].unsqueeze(1) + trans_logp).logsumexp(dim=1)
                # \sum_{z_{t-1}} p(z_{t-1}|x_{<t}) * p(z_t|z_{t-1}) * p(x_t|z_t)
            
        return -logp.sum(), (logp.unsqueeze(0) + trans_logp).logsumexp(dim=1)
    
    
    def sample_hmm(self, timesteps, logp_0, trials=1, cond_nll=None, viterbi=False):
        """
        Sample from the HMM latent variables.
        Forward-backward algorithm can perform smoothing estimate, filter is with forward algorithm.
        
        :param torch.Tensor cond_data: conditioning on data via nll -log p(x|z), of shape (maps, time)
        :param bool viterbi: return the MAP if true, otherwise conventional sampling
        :returns: state tensor of shape (trials, time)
        :rtype: torch.tensor
        """
        state = torch.empty((trials, timesteps), device=self.dummy.device).long()
        cc = 0
        
        iterator = tqdm(range(timesteps), leave=False) # AR sampling
        for t in iterator:
            if cond_nll is not None:
                cc = cond_nll[None, :, t]
            
            if t == 0:
                p = (logp_0[None, :] + cc).expand(trials, self.maps)
            else:
                p = torch.log(self.hmm_T[:, state[:, t-1]].T + 1e-12) + cc
                
            if viterbi:
                state[:, t] = torch.argmax(p, dim=1)
            else:
                s = dist.Multinomial(logits=p)()
                state[:, t] = s.topk(1)[1][:, 0].long()
            
        return state
    
    
    def objective(self, nll):
        """
        Compute the marginalized likelihood over discrete states.
        """
        if self.hmm_obs:
            time = torch.arange(nll.shape[1]).to(state.device)
            nll_term = nll[self.hmm_state[b].to(d_.device), time].sum()
        else: # HMM inference
            if b == 0:
                lp = self.logp_0
            else: # continuation between batches with graph cut
                lp = self.logp_hmm

            if enumerate_z: # marginalize the latent z, direct ML fitting, E-step
                nll_term, lp_0 = self.forward_hmm(nll, lp)
                self.logp_hmm = lp_0.data # cut from computational graph
            else: # Welch-Baum algorithm, M-step
                state = self.sample_hmm(nll.shape[1], lp, trials=1, cond_nll=nll, viterbi=True)
                time = torch.arange(state.shape[0]).to(state.device)
                nll_term = nll[state, time].sum() # use MAP latents

                D = self.maps-1
                self.logp_hmm = -12.*torch.ones(self.maps).to(state.device) # set starting point for next batch
                self.logp_hmm[state[-1]] = np.log(1.-D*1e-12)
        
        