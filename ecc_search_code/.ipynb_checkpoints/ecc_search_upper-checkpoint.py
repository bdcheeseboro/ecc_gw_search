#!/usr/bin/env python
# coding: utf-8

#Eccentric Residual Search

#This script runs a targeted upper limit eccentric search on a full PTA dataset.
# 
# Updated: 06/11/2021

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
from enterprise_extensions.blocks import (white_noise_block, red_noise_block, common_red_noise_block)
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
        elif 'equad' in line:
            par = 'log10_equad'
            flag = ln[0].split('equad-')[-1]
        elif 'jitter_q' in line:
            par = 'log10_ecorr'
            flag = ln[0].split('jitter_q-')[-1]
        elif 'RN-Amplitude' in line:
            par = 'red_noise_log10_A'
            flag = ''
        elif 'RN-spectral-index' in line:
            par = 'red_noise_gamma'
            flag = ''
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
    
    #separate pdist and pphase params
    pdist_params = [ p for p in params if 'p_dist' in p ]
    pphase_params = [ p for p in params if 'pphase' in p ]
    gammap_params = [ p for p in params if 'gamma_P' in p ]
    groups.extend([[params.index(pd) for pd in pdist_params]])
    groups.extend([[params.index(pp) for pp in pphase_params]])
    groups.extend([[params.index(gp) for gp in gammap_params]])
    
    if 'red noise' in params:

        # create parameter groups for the red noise parameters
        rnpsrs = [ p.split('_')[0] for p in params if '_log10_A' in p and 'gwb' not in p]
        b = [params.index(p) for p in params if 'alpha' in p]
        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])

        b = [params.index(p) for p in params if 'alpha' in p]
        groups.extend([b])

        for alpha in b:
            groups.extend([[alpha, params.index('J0613-0200_red_noise_gamma'), params.index('J0613-0200_red_noise_log10_A')]])


        for i in np.arange(0,len(b),2):
            groups.append([b[i],b[i+1]])


        groups.extend([[params.index(p) for p in rnpars]])
        a = [params.index(p) for p in rnpars]

    if 'gwb_log10_A' in params and 'gwb_gamma' in params:
        a.append(params.index('gwb_log10_A'))
        a.append(params.index('gwb_gamma'))
        if 'gwb_log10_fbend' in params:
            a.append(params.index('gwb_log10_fbend'))

        groups.extend([a])
    
    if 'e0' in pta.params:
        gpars = ['log10_Mc', 'e0', 'q', 'gamma0', 'l0', 'psi'] #global params
        groups.append([params.index(gp) for gp in gpars]) #add global params

        #pair global params
        groups.extend([[params.index('log10_Mc'), params.index('q')]])
        groups.extend([[params.index('log10_Mc'), params.index('e0')]])
        groups.extend([[params.index('gamma0'), params.index('l0')]])
        groups.extend([[params.index('gamma0'), params.index('psi')]])
        groups.extend([[params.index('psi'), params.index('l0')]])
        

        for pd, pp, gp in zip(pdist_params, pphase_params, gammap_params):
            groups.extend([[params.index(pd), params.index(pp), params.index(gp)]])
            groups.extend([[params.index(pd), params.index(pp), params.index(gp), params.index('log10_Mc')]])
            groups.extend([[params.index(pd), params.index(pp), params.index(gp), params.index('log10_Mc'), params.index('e0'), params.index('q')]])
    
    #parameters to catch and match gwb signals - if set to constant or not included, will skip

    crn_pars = ['gwb_gamma', 'gwb_log10_A']
    crn_cw_pars = crn_pars.copy()
    crn_cw_pars.extend(cw_pars)
    bpl_pars = ['gwb_gamma', 'gwb_log10_A', 'gwb_log10_fbend']
    bpl_cw_pars = bpl_pars.copy()
    bpl_cw_pars.extend(cw_pars)

    groups1 = []

    for pars in [crn_pars, crn_cw_pars, bpl_pars, bpl_cw_pars]:
        if any(item in params for item in pars):
            groups1.append(group_from_params(pta, pars))

    for group in groups1:
        if any(params.index(item) in group for item in amp_pars):
            pass
        else:
            for p in amp_pars:
                if p in params:
                    g = group.copy()
                    g.append(params.index(p))
                    groups1.append(g)

    groups.extend(groups1)
    
    return groups

#####
##  ARGUMENT PARSER
#####
parser = argparse.ArgumentParser(description='detection run searching for an ecc signal in a full PTA dataset')

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
parser.add_argument('-i','--inc',
                    action='store', required=True, type=float,
                    help='inclination of the binary orbital plane [rad]')
parser.add_argument('-f','--f_orb',
                    action='store', required=True, type=float,
                    help='orbital frequency [log10]')
parser.add_argument('-c','--crn',
                    action='store', required=True, type=bool,
                    help='add common red noise')
parser.add_argument('-d', '--datadir',
                    action='store', required=False, default='.',
                    help='directory containing the dataset')
parser.add_argument('-n', '--noisedir',
                    action='store', required=False, default='.',
                    help='directory containing the noise file(s)')
parser.add_argument('-o', '--outdir',
                    action='store', required=False, default='.',
                    help='polarization of the GW')
parser.add_argument('--pkl',
                    action='store', required=True, default='ideal_pulsars_ecc_search.pkl',
                    help='pickle file name')
args = parser.parse_args()

VERBOSE = args.verbose

#Simulated dataset directory path
datadir = args.datadir
noisepath = args.noisedir

#if there's a pickle file then use that
filename = datadir + args.pkl
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
    pickle.dump(psrs, open(datadir+args.pkl, 'wb'))

#add noise parameters
nfile = noisepath+'channelized_12p5yr_v3_full_noisedict.json'
if os.path.exists(nfile):
    with open(nfile) as nf:
        noise_params = json.load(nf)
else:
    noisefiles = sorted(glob.glob(noisepath+'/*.txt'))
    noise_params = {}
    for nf in noisefiles:
        noise_params.update(get_noise_from_pal2(nf))

# white noise parameters
# set them to constant here and we will input the noise values after the model is initialized
efac = parameter.Constant()
equad = parameter.Constant()
ecorr = parameter.Constant()

# define selection by observing backend
selection = selections.Selection(selections.by_backend)

# define white noise signals
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection, name='')

# red noise
rn = red_noise_block(prior='log-uniform')
common = args.crn
if common:
    crn = common_red_noise_block(prior='log-uniform', name='gwb')

#Eccentric gw parameters
#gw parameters
gwphi = parameter.Constant(args.gwphi)('gwphi') #RA of source
gwtheta = parameter.Constant(args.gwtheta)('gwtheta') #DEC of source
log10_dist = parameter.Constant(args.gwdist)('log10_dist') #distance to source

#orbital parameters
log10_forb = parameter.Constant(args.forb)('log10_forb') #log10 orbital frequency
inc = parameter.Constant(args.inc)('inc') #inclination of the binary's orbital plane

#Search parameters
#when searching over pdist there is no need to search over pphase bcuz the code
#calculates the phase for that pulsar.
q = parameter.LinearExp(0.1,1)('q') #mass ratio
log10_mc = parameter.LinearExp(7,11)('log10_Mc') #log10 chirp mass
e0 = parameter.LinearExp(0.001, 0.99)('e0') #eccentricity
p_dist = parameter.Normal(0,1) #prior on pulsar distance
pphase = parameter.Uniform(0,2*np.pi) #prior on pulsar phase
gamma_P = parameter.Uniform(0,2*np.pi) #prior on pulsar gamma
l0 = parameter.Uniform(0,2*np.pi)('l0') #mean anomaly
gamma0 = parameter.Uniform(0,2*np.pi)('gamma0') #initial angle of periastron
psi = parameter.Uniform(0,2*np.pi)('psi') #polarization of the GW

#Calcuate tref based on latest TOA MJd
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
tref = max(tmax)/86400

#Eccentric signal construction
#To create a signal to be used by enterprise you must first create a residual 
#and use CWSignal to convert the residual as part of the enterprise Signal class
ewf = ecc_res.add_ecc_cgw(gwtheta=gwtheta, gwphi=gwphi, log10_mc=log10_mc, q=q, log10_forb=log10_forb, e0=e0, l0=l0, gamma0=gamma0, 
                    inc=inc, psi=psi, log10_dist=log10_dist, p_dist=p_dist, pphase=pphase, gamma_P=gamma_P, tref=tref, 
                    psrterm=True, evol=True, waveform_cal=True, res='Both')
ew = CWSignal(ewf, ecc=False, psrTerm=False)

# linearized timing model
tm = gp_signals.TimingModel(use_svd=False)

# full signal with red noise and white noise signals
s = ef + tm + ew + rn + eq + ec

if common:
    s += crn

# initialize PTA
model = [s(psr) for psr in psrs]
pta = signal_base.PTA(model)

pta.set_default_params(noise_params)

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
pdist_params = [psr.name+'_cw_p_dist' for psr in psrs]
for pd in pdist_params:
    sampler.addProposalToCycle(jp.draw_from_par_prior(pd),5)

#draw from phase priors
pphase_params = [psr.name+'_cw_pphase' for psr in psrs]
for pp in pphase_params
    sampler.addProposalToCycle(jp.draw_from_par_prior(pp),5)

#draw from gamma_P priors
gammap_params = [psr.name+'_cw_gamma_P' for psr in psrs]
for gp in gammap_params
    sampler.addProposalToCycle(jp.draw_from_par_prior(gp),5)

rn_params = [psr.name+'_red_noise_gamma' for psr in psrs]
for rnp in rn_params:
    sampler.addProposalToCycle(jp.draw_from_par_prior(rnp),5)

rna_params = [psr.name+'_red_noise_log10_A' for psr in psrs]
for rna in rna_params:
    sampler.addProposalToCycle(jp.draw_from_par_prior(rna),5)

if common:
    crn_params = ['gwb_log10_A', 'gwb_gamma']
    for crn in crn_params:
        sampler.addProposalToCycle(jp.draw_from_par_prior(crn),5)

#draw from ewf priors
ew_params = ['e0','log10_mc', 'q', 'l0', 'gamma0', 'psi']
for ew in ew_params:
    sampler.addProposalToCycle(jp.draw_from_par_prior(ew),5)

N = int(4.5e6)
sampler.sample(xecc, N, SCAMweight=50, AMweight=50, DEweight=0)