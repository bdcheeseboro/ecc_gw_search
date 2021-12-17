from __future__ import division
import glob
import os
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

from scipy.stats import skewnorm

from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils

from enterprise_extensions.model_utils import EmpiricalDistribution1D
from enterprise_extensions.model_utils import EmpiricalDistribution2D

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


def get_global_parameters(pta):
    """Utility function for finding global parameters."""
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)
    
    gpars = np.unique(list(filter(lambda x: pars.count(x)>1, pars)))
    ipars = np.array([p for p in pars if p not in gpars])

    return gpars, ipars


def get_parameter_groups(pta):
    """Utility function to get parameter groupings for sampling."""
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names

    snames = np.unique([[qq.signal_name for qq in pp._signals] 
                        for pp in pta._signalcollections])
    
    # sort parameters by signal collections
    ephempars = []
    rnpars = []
    cwpars = []

    for sc in pta._signalcollections:
        for signal in sc._signals:
            if signal.signal_name == 'red noise':
                rnpars.extend(signal.param_names)
            elif signal.signal_name == 'phys_ephem':
                ephempars.extend(signal.param_names)
            elif signal.signal_name == 'cw':
                cwpars.extend(signal.param_names)
    
    if 'red noise' in snames:
    
        # create parameter groups for the red noise parameters
        rnpsrs = [ p.split('_')[0] for p in params if '_log10_A' in p ]

        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_red_noise_gamma'), 
                            params.index(psr + '_red_noise_log10_A')]])
                    
    # set up groups for the BayesEphem parameters
    if 'phys_ephem' in snames:
        
        ephempars = np.unique(ephempars)
        juporb = [p for p in ephempars if 'jup_orb' in p]
        groups.extend([[params.index(p) for p in ephempars if p not in juporb]])
        groups.extend([[params.index(jp) for jp in juporb]])
        for i1 in range(len(juporb)):
            for i2 in range(i1+1, len(juporb)):
                groups.extend([[params.index(p) for p in [juporb[i1], juporb[i2]]]])
        
    if 'cw' in snames:
    
        # divide the cgw parameters into two groups: 
        # the common parameters and the pulsar phase and distance parameters
        cw_common = np.unique(list(filter(lambda x: cwpars.count(x)>1, cwpars)))
        groups.extend([[params.index(cwc) for cwc in cw_common]])

        cw_pulsar = np.array([p for p in cwpars if p not in cw_common])
        if len(cw_pulsar) > 0:
            
            pdist_params = [ p for p in cw_pulsar if 'p_dist' in p ]
            pphase_params = [ p for p in cw_pulsar if 'p_phase' in p ]
            
            groups.extend([[params.index(pd) for pd in pdist_params]])
            groups.extend([[params.index(pp) for pp in pphase_params]])
            
            for pd,pp in zip(pdist_params,pphase_params):
                groups.extend([[params.index(pd), params.index(pp)]])
                groups.extend([[params.index(pd), params.index(pp), params.index('cw_costheta')]])
                groups.extend([[params.index(pd), params.index(pp), params.index('cw_phi')]])
                groups.extend([[params.index(pd), params.index(pp), params.index('cw_phase0')]])
                groups.extend([[params.index(pd), params.index(pp), params.index('cw_log10_Mc')]])
                groups.extend([[params.index(pd), params.index(pp), params.index('cw_log10_h')]])
                groups.extend([[params.index(pd), params.index(pp), params.index('cw_cosinc')]])
                groups.extend([[params.index(pd), params.index(pp), params.index('cw_psi')]])
                groups.extend([[params.index(pd), params.index('cw_costheta'), params.index('cw_phi')]])
                groups.extend([[params.index(pd), params.index('cw_log10_h'), params.index('cw_log10_Mc')]])
                groups.extend([[params.index(pp), params.index('cw_costheta'), params.index('cw_phi')]])
                groups.extend([[params.index(pp), params.index('cw_phase0'), params.index('cw_log10_Mc'), 
                                params.index('cw_cosinc'), params.index('cw_psi')]])
                groups.extend([[params.index(pd), params.index(pp), 
                                params.index('cw_log10_Mc'), params.index('cw_costheta'), params.index('cw_phi')]])
                
                if 'red noise' in snames:
                    psr = pd.split('_')[0]
                    groups.extend([[params.index(pd), params.index(pp), 
                                    params.index(psr + '_red_noise_gamma'), 
                                    params.index(psr + '_red_noise_log10_A')]])
                        
        # now try other combinations of the common cgw parameters
        combos = [['cw_costheta', 'cw_phi'], 
                  ['cw_costheta', 'cw_phi', 'cw_log10_h'], 
                  ['cw_costheta', 'cw_phi', 'cw_psi'], 
                  ['cw_log10_h', 'cw_cosinc', 'cw_phase0', 'cw_psi'], 
                  ['cw_log10_Mc', 'cw_phase0'], 
                  ['cw_cosinc', 'cw_phase0'],
                  ['cw_phase0', 'cw_psi'], 
                  ['cw_costheta', 'cw_log10_h'], 
                  ['cw_cosinc', 'cw_log10_h'], 
                  ['cw_cosinc', 'cw_psi'], 
                  ['cw_phi', 'cw_log10_h'], 
                  ['cw_log10_h', 'cw_log10_Mc'], 
                  ['cw_log10_h', 'cw_phase0'], 
                  ['cw_log10_h', 'cw_psi']]
        for combo in combos:
            if all(c in cw_common for c in combo):
                groups.extend([[params.index(c) for c in combo]])
                
    if 'cw' in snames and 'phys_ephem' in snames:
        # add a group that contains the Jupiter orbital elements and the common GW parameters
        myparams = [p for p in params if p in juporb or p in cw_common]
        
        groups.extend([[params.index(p) for p in myparams]])
                    
    return groups


class JumpProposal(object):
    
    def __init__(self, pta, snames=None, fgw=3e-8, psr_dist=None, 
                 rnposteriors=None, juporbposteriors=None):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.npar = len(pta.params)
        self.ndim = sum(p.size or 1 for p in pta.params)
        self.psrnames = pta.pulsars
        
        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size
        
        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct
        
        # collecting signal parameters across pta
        if snames is None:
            allsigs = np.hstack([[qq.signal_name for qq in pp._signals]
                                 for pp in pta._signalcollections])
            self.snames = dict.fromkeys(np.unique(allsigs))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = list(set(self.snames[key]))
        else:
            self.snames = snames
            
        self.fgw = fgw
        self.psr_dist = psr_dist

        # initialize empirical distributions for the red noise parameters from a previous MCMC
        if rnposteriors is not None and os.path.isfile(rnposteriors):
            with open(rnposteriors) as f:
                self.rnDistr = pickle.load(f)
                self.rnDistr_psrnames = [ r.param_names[0].split('_')[0] for r in self.rnDistr ]
        else:
            self.rnDistr = None
            self.rnDistr_psrnames = None

        # initialize empirical distributions for the Jupiter orbital elements from a previous MCMC
        if juporbposteriors is not None and os.path.isfile(juporbposteriors):
            with open(juporbposteriors) as f:
                self.juporbDistr = pickle.load(f)
        else:
            self.juporbDistr = None

    def draw_from_ephem_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        signal_name = 'phys_ephem'
        
        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) \
                    - param.get_logpdf(q[self.pmap[str(param)]])
    
        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()
            
            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) \
                    - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)

    def draw_from_jup_orb_priors(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == 'jup_orb_elements':
                idx = i
        
        # draw parameter from signal model
        param = self.params[idx]
        idx2 = np.random.randint(0, param.size)
            
        q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) \
                    - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)
    
    def draw_from_jup_orb_1d_posteriors(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        if self.juporbDistr is not None:
            
            # randomly choose one of the 1D posteriors of the Jupiter orbital elements
            j = np.random.randint(0, len(self.juporbDistr))
        
            idx = self.pimap[str(self.juporbDistr[j].param_name)]
        
            q[idx] = self.juporbDistr[j].draw()
            lqxy = self.juporbDistr[j].logprob(x[idx]) - self.juporbDistr[j].logprob(q[idx])

        else:
            
            # if there is no empirical distribution for the Jupiter orbital elements,
            # draw from the prior instead
            idx = 0
            for i,p in enumerate(self.params):
                if p.name == 'jup_orb_elements':
                    idx = i
        
            # draw parameter from signal model
            param = self.params[idx]
            idx2 = np.random.randint(0, param.size)
            
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) \
                    - param.get_logpdf(q[self.pmap[str(param)]])
                    
        return q, float(lqxy)
    
    def draw_skyposition(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        # jump in both cos_gwtheta and gwphi, drawing both values from the prior
        cw_params = ['gwtheta', 'gwphi']
        for cp in cw_params:
            
            idx = 0
            for i,p in enumerate(self.params):

                if p.name == cp:
                    idx = i
        
            # draw parameter from signal model
            param = self.params[idx]
            q[self.pmap[str(param)]] = param.sample()
        
        return q, float(lqxy)

    def draw_from_cw_log_uniform_distribution(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        idx = self.pnames.index('cw_log10_h')
        q[idx] = np.random.uniform(-18, -11)
        
        return q, float(lqxy)

    def draw_from_cw_prior(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        # randomly choose one of the cw parameters and get index
        cw_params = [ p for p in self.pnames if p in ['gwtheta', 'gwphi', 'q', 'log10_forb','log10_Mc', 
                                                      'e0']]
        myparam = np.random.choice(cw_params)
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i
        
        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)

    
    def draw_from_rnposteriors(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        mypsr = np.random.choice(self.psrnames)
        
        if self.rnDistr is not None and mypsr in self.rnDistr_psrnames:
                
            i = self.rnDistr_psrnames.index(mypsr)
            
            oldsample = [x[self.pnames.index(mypsr + '_red_noise_gamma')], 
                         x[self.pnames.index(mypsr + '_red_noise_log10_A')]]
                
            newsample = self.rnDistr[i].draw() 
            
            q[self.pnames.index(mypsr + '_red_noise_gamma')] = newsample[0]
            q[self.pnames.index(mypsr + '_red_noise_log10_A')] = newsample[1]
            
            # forward-backward jump probability
            lqxy = self.rnDistr[i].logprob(oldsample) - self.rnDistr[i].logprob(newsample)
                
        else:

            # if there is no empirical distribution for this pulsar's red noise parameters, 
            # choose one of the red noise parameters and draw a sample from the prior
            myparam = np.random.choice([mypsr + '_red_noise_gamma', mypsr + '_red_noise_log10_A'])
            idx = 0
            for i,p in enumerate(self.params):

                if p.name == myparam:
                    idx = i

            # draw parameter from signal model
            param = self.params[idx]
            q[self.pmap[str(param)]] = param.sample()
        
            # forward-backward jump probability
            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)
    
    def draw_from_rnpriors(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        mypsr = np.random.choice(self.psrnames)
        
        myparam = np.random.choice([mypsr + '_red_noise_gamma', mypsr + '_red_noise_log10_A'])
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i

        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        return q, float(lqxy)
    
    def draw_from_pdist_prior(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        myparam = np.random.choice([p for p in self.pnames if 'p_dist' in p])
        
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i

        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)

    def draw_from_pphase_prior(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        myparam = np.random.choice([p for p in self.pnames if 'p_phase' in p])
        
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i

        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)
    
    def draw_strain_skewstep(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        a = 2
        s = 1
        
        diff = skewnorm.rvs(a, scale=s)
        q[self.pnames.index('cw_log10_h')] = x[self.pnames.index('cw_log10_h')] - diff
        lqxy = skewnorm.logpdf(-diff, a, scale=s) - skewnorm.logpdf(diff, a, scale=s)
        
        return q, float(lqxy)

    def draw_strain_inc(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # half of the time, jump so that you conserve h*(1 + cos_inc^2)
        # the rest of the time, jump so that you conserve h*cos_inc
        
        which_jump = np.random.random()
        
        if which_jump > 0.5:
        
            q[self.pnames.index('cw_cosinc')] = np.random.uniform(-1,1)
            q[self.pnames.index('cw_log10_h')] = x[self.pnames.index('cw_log10_h')] \
                                                + np.log10(1+x[self.pnames.index('cw_cosinc')]**2) \
                                                - np.log10(1+q[self.pnames.index('cw_cosinc')]**2)
                    
        else:
            
            # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
            if x[self.pnames.index('cw_cosinc')] > 0:
                q[self.pnames.index('cw_cosinc')] = np.random.uniform(0,1)
            else:
                q[self.pnames.index('cw_cosinc')] = np.random.uniform(-1,0)

            q[self.pnames.index('cw_log10_h')] = x[self.pnames.index('cw_log10_h')] \
                                                + np.log10(x[self.pnames.index('cw_cosinc')]/q[self.pnames.index('cw_cosinc')])
                
        return q, float(lqxy)
    
    def draw_strain_psi(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # draw a new value of psi, then jump in log10_h so that either h*cos(2*psi) or h*sin(2*psi) are conserved
        which_jump = np.random.random()
        
        if which_jump > 0.5:
            # jump so that h*cos(2*psi) is conserved            
            # make sure that the sign of cos(2*psi) does not change
            if x[self.pnames.index('cw_psi')] > 0.25*np.pi and x[self.pnames.index('cw_psi')] < 0.75*np.pi:
                q[self.pnames.index('cw_psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
            else:
                newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                if newval < 0:
                    newval += np.pi
                q[self.pnames.index('cw_psi')] = newval
                
            ratio = np.cos(2*x[self.pnames.index('cw_psi')])/np.cos(2*q[self.pnames.index('cw_psi')])
            q[self.pnames.index('cw_log10_h')] = x[self.pnames.index('cw_log10_h')] + np.log10(ratio)       
            
        else:
            # jump so that h*sin(2*psi) is conserved            
            # make sure that the sign of sin(2*psi) does not change
            if x[self.pnames.index('cw_psi')] < np.pi/2:
                q[self.pnames.index('cw_psi')] = np.random.uniform(0,np.pi/2)
            else:
                q[self.pnames.index('cw_psi')] = np.random.uniform(np.pi/2,np.pi)
                
            ratio = np.sin(2*x[self.pnames.index('cw_psi')])/np.sin(2*q[self.pnames.index('cw_psi')])
            q[self.pnames.index('cw_log10_h')] = x[self.pnames.index('cw_log10_h')] + np.log10(ratio)
                
        return q, float(lqxy)
    
    def phase_psi_reverse_jump(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0

        param = np.random.choice([str(p) for p in self.pnames if 'phase' in p])
        
        if param == 'cw_phase0':
            q[self.pnames.index('cw_phase0')] = np.mod(x[self.pnames.index('cw_phase0')] + np.pi, 2*np.pi)
            q[self.pnames.index('cw_psi')] = np.mod(x[self.pnames.index('cw_psi')] + np.pi/2, np.pi)
        else:
            q[self.pnames.index(param)] = np.mod(x[self.pnames.index(param)] + np.pi, 2*np.pi)
                
        return q, float(lqxy)
    
    def draw_mc_pphase_pdist(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # draw a small step in log10_mc
        q[self.pnames.index('cw_log10_Mc')] = x[self.pnames.index('cw_log10_Mc')] + 0.1*np.random.randn()
        
        # now adjust the pulsar phases and distances
        ratio = 10**((5/3)*(x[self.pnames.index('cw_log10_Mc')]-q[self.pnames.index('cw_log10_Mc')]))
        
        mypsr = np.random.choice(self.psrnames)
        q[self.pnames.index(mypsr + '_cw_p_phase')] = np.mod(x[self.pnames.index(mypsr + '_cw_p_phase')]*ratio, 2*np.pi)
        
        q[self.pnames.index(mypsr + '_cw_p_dist')] = ratio*x[self.pnames.index(mypsr + '_cw_p_dist')] \
                                                        + (ratio-1.)*(self.psr_dist[mypsr][0]/self.psr_dist[mypsr][1])
        
        return q, float(lqxy)
    
    def fix_cyclic_pars(self, prepar, postpar, iter, beta):
        
        q = postpar.copy()
        
        for param in self.params:
            if 'phase' in param.name:
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
            elif param.name == 'cw_psi':
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], np.pi)
            elif param.name == 'cw_phi':
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
                
        return q, 0
