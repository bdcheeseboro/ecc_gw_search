#!/usr/bin/env python
# coding: utf-8

#Eccentric Residual Search Ideal

# This script tests on a simulated dataset containing an eccentric gw signal. Based on work done by Sarah Vigeland, Ph.D. from `cw_search_sample.ipynb`
# 
# Updated: 04/01/2021

from __future__ import division
import numpy as np
import glob
import os
import pickle
import json
import matplotlib.pyplot as plt
import corner
import sys
import argparse

from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils
from enterprise_extensions.deterministic import CWSignal
from enterprise.signals.signal_base import SignalCollection
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions.sampler import JumpProposal as JP
import arviz as az
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
import ecc_res
import scipy.constants as sc

def get_noise_from_pal2(noisefile):
    psrname = noisefile.split('/')[-1].split('_noise.txt')[0]
    fin = open(noisefile, 'r')
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if 'efac' in line:
            par = 'efac'
            flag = ln[0].split('efac-')[-1]
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = '_'.join(name)
        params.update({pname: float(ln[1])})
    return params

def get_ew_groups(pta):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    params = pta.param_names
    ndim = len(params)
    groups = [list(np.arange(0, ndim))]
    gpars = ['log10_Mc', 'log10_forb', 'e0', 'q'] #global params
    groups.append([params.index(gp) for gp in gpars]) #add global params
    
    #pair global params
    groups.extend([[params.index('log10_Mc'), params.index('log10_forb')]])
    groups.extend([[params.index('log10_Mc'), params.index('q')]])
    groups.extend([[params.index('log10_Mc'), params.index('e0')]])
    groups.extend([[params.index('e0'), params.index('log10_forb')]])
    
    #separate pdist and pphase params
    pdist_params = [ p for p in params if 'p_dist' in p ]
    pphase_params = [ p for p in params if 'pphase' in p ]
    gammap_params = [ p for p in params if 'gamma_P' in p ]
    groups.extend([[params.index(pd) for pd in pdist_params]])
    groups.extend([[params.index(pp) for pp in pphase_params]])
    groups.extend([[params.index(gp) for gp in gammap_params]])
    
    for pd, pp, gp in zip(pdist_params, pphase_params, gammap_params):
        groups.extend([[params.index(pd), params.index(pp), params.index(gp)]])
        groups.extend([[params.index(pd), params.index(pp), params.index(gp), params.index('log10_Mc'), params.index('log10_forb')]])
        groups.extend([[params.index(pd), params.index(pp), params.index(gp), params.index('log10_Mc'), params.index('log10_forb'), params.index('e0'), params.index('q')]])
    
    return groups

#####
##  ARGUMENT PARSER
#####
parser = argparse.ArgumentParser(description='run an MCMC on three bursts using wavelets and eccprior')

parser.add_argument('-v', '--verbose',
                    action='store_true', required=False,
                    help='print verbose output')
parser.add_argument('--gwphi',
                    action='store', required=True, type=float,
                    help='RA of source [rad]')
parser.add_argument('--gwtheta',
                    action='store', required=True, type=float,
                    help='DEC of source [rad]')
parser.add_argument( '--gwdist',
                    action='store', required=True, type=float,
                    help='log10 based distance to source [pc]')
parser.add_argument('-l','--l0',
                    action='store', required=True, type=int,
                    help='mean anomaly at reference time [rad]')
parser.add_argument('-g','--gamma0',
                    action='store', required=True, type=float,
                    help='initial angle of periastron [rad]')
parser.add_argument('-i','--inc',
                    action='store', required=True, type=float,
                    help='inclination of the binary orbital plane [rad]')
parser.add_argument('-p','--psi',
                    action='store', required=True, type=float,
                    help='polarization of the GW')
parser.add_argument('-d', '--datadir',
                    action='store', required=False, default='.',
                    help='directory containing the dataset')
parser.add_argument('-n', '--noisedir',
                    action='store', required=False, default='.',
                    help='directory containing the noise file(s)')
parser.add_argument('-o', '--outdir',
                    action='store', required=False, default='.',
                    help='polarization of the GW')

args = parser.parse_args()

VERBOSE = args.verbose

#Simulated dataset directory path
datadir = args.datadir
noisepath = args.noisedir

#get noise files
#noisefiles = sorted(glob.glob(noisepath+'/*.txt'))

#if there's a pickle file then use that
filename = datadir + 'ideal_pulsars_ecc_search.pkl'
if os.path.exists(filename):
    with open(filename, "rb") as f:
        psrs = pickle.load(f)
#else load the par and tim files in and make a pickle file for the future
else:
    psrs = []
    #load par, tim, and noise files for each of the pulsars
    parfiles = sorted(glob.glob(datadir+'/*.par'))
    timfiles = sorted(glob.glob(datadir+'/*.tim'))
    for p, t in zip(parfiles, timfiles):
        print('Loading pulsar from parfile {0}'.format(p))
        psrs.append(Pulsar(p, t))
    pickle.dump(psrs, open(datadir+'ideal_pulsars_ecc_search.pkl', 'wb'))

#grab noise json file
with open(noisepath+'channelized_12p5yr_v3_full_noisedict.json') as nf:
    noise_params = json.load(nf)

# white noise parameter
efac = parameter.Constant()
selection = selections.Selection(selections.by_backend)
# white noise signal
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

#Eccentric gw parameters
#gw parameters
gwphi = parameter.Constant(args.gwphi)('gwphi') #RA of source
gwtheta = parameter.Constant(args.gwtheta)('gwtheta') #DEC of source
log10_dist = parameter.Constant(args.gwdist)('log10_dist') #distance to source

#orbital parameters
l0 = parameter.Constant(args.l0)('l0') #mean anomaly
gamma0 = parameter.Constant(args.gamma0)('gamma0') #initial angle of periastron
inc = parameter.Constant(args.inc)('inc') #inclination of the binary's orbital plane
psi = parameter.Constant(args.psi)('psi') #polarization of the GW

#Search parameters
#when searching over pdist there is no need to search over pphase bcuz the code
#calculates the phase for that pulsar.
q = parameter.Uniform(0.1,1)('q') #mass ratio
log10_mc = parameter.Uniform(7,11)('log10_Mc') #log10 chirp mass
e0 = parameter.Uniform(0.001, 0.1)('e0') #eccentricity
log10_forb = parameter.Uniform(-9,-7)('log10_forb') #log10 orbital frequency
p_dist = parameter.Normal(0,1) #prior on pulsar distance
pphase = parameter.Uniform(0,2*np.pi) #prior on pulsar phase


#Calcuate tref based on latest TOA MJd
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
tref = max(tmax)/86400

#Eccentric signal construction
#To create a signal to be used by enterprise you must first create a residual 
#and use CWSignal to convert the residual as part of the enterprise Signal class
ewf = ecc_res.add_ecc_cgw(gwtheta=gwtheta, gwphi=gwphi, log10_mc=log10_mc, q=q, log10_forb=log10_forb, e0=e0, l0=l0, gamma0=gamma0, 
                    inc=inc, psi=psi, log10_dist=log10_dist, p_dist=p_dist, pphase=pphase, gamma_P=None, tref=tref, 
                    psrterm=True, evol=True, waveform_cal=True, res='Both')
ew = CWSignal(ewf, ecc=False, psrTerm=False)

# linearized timing model
tm = gp_signals.TimingModel(use_svd=False)
# full signal (no red noise added at this time)
s = ef + tm + ew

# initialize PTA
model = [s(psr) for psr in psrs]
pta = signal_base.PTA(model)

#add noise parameters to the pta object
#params = {}
#for nf in noisefiles:
#    params.update(get_noise_from_pal2(nf))

#extract efac params and set noise params
noise_dict = {}
for psr in psrs:
    noise_dict[psr.name]={}
    noise_dict[psr.name]['efacs'] = []
    for ky in list(noise_params.keys()):
        if psr.name in ky:
            if 'efac' in ky:
                noise_dict[psr.name]['efacs'].append([ky.replace(psr.name + '_' , ''), noise_params[ky]])
    noise_dict[psr.name]['efacs'] = np.array(noise_dict[psr.name]['efacs'])
pta.set_default_params(noise_dict)

#Select sample from the search parameters
xecc = np.hstack(np.array([p.sample() for p in pta.params]))
ndim = len(xecc)

# initialize pulsar distance parameters
p_dist_params = [ p for p in pta.param_names if 'p_dist' in p ]
for pd in p_dist_params:
    xecc[pta.param_names.index(pd)] = 0

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

groups = get_ew_groups(pta)

#output directory for all the chains, params, and groups
chaindir = args.outdir

#Setup sampler
resume = True
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                 outDir=chaindir, resume=resume)

# write parameter file and parameter groups file
np.savetxt(chaindir + 'params.txt', list(map(str, pta.param_names)), fmt='%s')
np.savetxt(chaindir + 'groups.txt', groups, fmt='%s')

# add prior draws to proposal cycle
jp = JP(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 5)

#draw from pdist priors
pdist_params = ['B1855+09_cw_p_dist', 'J0030+0451_cw_p_dist', 'J0613-0200_cw_p_dist','J1012+5307_cw_p_dist', 'J1024-0719_cw_p_dist', 'J1455-3330_cw_p_dist', 'J1600-3053_cw_p_dist', 'J1640+2224_cw_p_dist', 'J1744-1134_cw_p_dist', 'J1909-3744_cw_p_dist']
sampler.addProposalToCycle(jp.draw_from_par_prior(pdist_params),5)

#draw from phase priors
pphase_params = ['B1855+09_cw_pphase', 'J0030+0451_cw_pphase', 'J0613-0200_cw_pphase','J1012+5307_cw_pphase', 'J1024-0719_cw_pphase', 'J1455-3330_cw_pphase', 'J1600-3053_cw_pphase', 'J1640+2224_cw_pphase', 'J1744-1134_cw_pphase', 'J1909-3744_cw_pphase']
sampler.addProposalToCycle(jp.draw_from_par_prior(pphase_params),5)

#draw from gamma_P priors
gammap_params = ['B1855+09_cw_gamma_P', 'J0030+0451_cw_gamma_P', 'J0613-0200_cw_gamma_P','J1012+5307_cw_gamma_P', 'J1024-0719_cw_gamma_P','J1455-3330_cw_gamma_P', 'J1600-3053_cw_gamma_P', 'J1640+2224_cw_gamma_P', 'J1744-1134_cw_gamma_P', 'J1909-3744_cw_gamma_P']
sampler.addProposalToCycle(jp.draw_from_par_prior(gammap_params),5)

#draw from ewf priors
ew_params = ['e0', 'log10_forb','log10_mc', 'q']
sampler.addProposalToCycle(jp.draw_from_par_prior(ew_params),10)

#draw from forb and mc priors
f_mc = ['log10_forb','log10_mc']
sampler.addProposalToCycle(jp.draw_from_par_prior(f_mc),10)


N = int(4.5e6)
sampler.sample(xecc, N, SCAMweight=50, AMweight=50, DEweight=0)