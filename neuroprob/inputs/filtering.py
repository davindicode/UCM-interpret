def particle_filtering(self):
    """
    Suitable for AR(1) priors.
    """
    def __init__(self, covariates, bounds, samples=1000):
        """
        Rejection sampling to get the posterior samples
        Dynamic spatial maps supported
        """
        #self.register_buffer('samples', torch.empty((samples, covariates)))
        #self.register_buffer('weights', torch.empty((samples)))
        self.sam_cnt = samples
        self.eff_cnt = samples
        self.samples = np.empty((samples, covariates))
        self.weights = np.empty((samples))
        self.bounds = bounds

        
    def check_bounds(self):
        for k in range(self.bounds.shape[0]):
            np.clip(self.samples[:, k], self.bounds[k, 0], self.bounds[k, 1], out=self.samples[:, k])   

            
    def initialize(self, mu, std):
        """
        Initial samples positions, uniform weights
        """
        self.samples = std*np.random.randn(*self.samples.shape) + mu
        self.check_bounds()
        self.weights.fill(1/self.sam_cnt)

        
    def predict(self, sigma):
        """
        Prior distribution propagation
        """
        gauss = np.random.randn(*self.samples.shape)
        self.samples += sigma*gauss
        self.check_bounds()

        
    def update(self, eval_rate, activity, tbin):
        """
        Use IPP likelihood for cognitive maps to assign weights
        TODO use NLL directly
        """
        units = len(activity)
        fact = factorial(activity)
        w = 1.
        for u in range(units):
            rT = tbin*eval_rate[u]((self.samples[:, 0], self.samples[:, 1], np.zeros_like(self.samples[:, 0])))
            if activity[u] > 0:
                w *= rT**activity[u] / fact[u] * np.exp(-rT)

        self.weights *= w
        self.weights += 1e-12
        self.weights /= self.weights.sum()

        self.eff_cnt = np.sum(np.square(self.weights))

        
    def resample(self):
        """
        Stratified resampling algorithm
        """
        positions = (np.random.rand(self.sam_cnt) + np.arange(self.sam_cnt)) / self.sam_cnt
        indexes = np.zeros(self.sam_cnt, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.sam_cnt:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.samples = self.samples[indexes, :]
        self.weights.fill(1/self.sam_cnt)

        
    def estimate(self):
        """
        Get moments of particle distribution
        """
        rew = self.samples*self.weights[:, None]
        mu = rew.sum(0)
        cov = np.cov(rew.T)
        return mu, cov
    

    
### causal decoding ###
def VI_filtering(model, ini_X, spikes, VI_steps, fitting_options, past_spikes=None):
    """
    Decode causally taking into account covariate prior. In the most general case, treat model as LVM 
    but fix the latent variables at all timesteps except the last and fit model with fixed tuning. Note 
    filtering has to be done recursively, simultaneous inference of LVM is smoothing.

    Past covariate values before the decoding window (relevant in e.g. GP priors) are inserted via ``ini_X``, 
    identical to the procedure of initialization with *preprocess()*. Observed covariates will be variational 
    distributions with zero standard deviation.

    All model parameters are kept constant, only the decoded variables are optimized. Optimizers need to 
    be set up before with *set_optimizers()*.

    Reduces to Kalman filtering when Gaussian and Euclidean.

    :param list ini_X: 
    :param np.array spikes: 
    :param np.array past_spikes: spikes before the decoding window (relevant for GLMs with filter_len>1)
    """
    assert self.maps == 1 # no HMM supported

    rc_t = np.concatenate((past_spikes, spikes), dim=-1) if past_spikes is not None else spikes
    decode_len = spikes.shape[-1]
    past_len = past_spikes.shape[-1] if past_spikes is not None else 0
    resamples = rc_t.shape[-1]

    # set all parameters to not learnable
    label_for_constrain = []
    for name, param in self.named_parameters():
        if name[:16] == 'inputs.lv_':
            label_for_constrain.append(param)
            continue
        param.requires_grad = False

    def func(obj):
        decode_past = obj.saved_covariates[0].shape[0]
        k = 0
        for name, param in obj.named_parameters():
            if name[:16] == 'inputs.lv_':
                param.data[:decode_past] = obj.saved_covariates[k].shape[0]
                k += 1

    for k in range(past_len+1, past_len+decode_len):
        bs = [k+1, resamples-k-1] if resamples-k-1 > 0 else k+1
        model.inputs.set_XZ(ini_X, resamples, rc_t, bs) # non-continous batches
        model.likelihood.set_Y(rc_t)
        
        model.batches = 1 # only optimize first batch
        model.saved_covariates = []
        for p in label_for_constrain:
            model.saved_covariates.append(p.data[:k-1])

        # infer only last LV in batch 0
        model.fit(VI_steps, *fitting_options, callback=func) #loss_margin=-1e2, margin_epochs=100, anneal_func=annealing, 
             #cov_samples=32, ll_samples=1, bound='ELBO', ll_mode='MC'

    # set all parameters to learnable
    for param in self.parameters():
        param.requires_grad = True

    del model.saved_covariates

    return



