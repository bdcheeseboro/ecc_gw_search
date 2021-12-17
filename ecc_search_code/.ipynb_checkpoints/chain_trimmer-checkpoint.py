from __future__ import division
import numpy as np
import glob, json
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy as sp
import os
import pickle as pickle

from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils

from enterprise.signals.gp_priors import t_process
from enterprise.signals.gp_priors import InvGamma, InvGammaPrior, InvGammaSampler

from enterprise_extensions import model_utils


import targeted_functions as fns

from operator import itemgetter
import itertools

import acor

def range_bad_indices(f_chain):
    range_tot = max(f_chain)-min(f_chain)

    bad_chunks = []
    for i in range(0,len(f_chain)-200):
        range_chunk = max(f_chain[i:i+200])-min(f_chain[i:i+200])
        if np.abs(range_chunk) < 0.15*range_tot:
            bad_chunks.append(i)

    for i in range(len(bad_chunks)-1):
        if bad_chunks[i+1]-bad_chunks[1]>1:
            for j in range(200):
                bad_chunks.append(bad_chunks[i]+j)
                
    return bad_chunks

def range_edges(bad_chunks):
    bad_start = []
    bad_end = []
    L = bad_chunks
    for k, g in itertools.groupby( enumerate(L), lambda x: x[1]-x[0] ) :
        a = list(map(itemgetter(1), g))
        bad_start.append(a[0])
        bad_end.append(a[-1]+200)

    edges_start = []
    edges_end = []
    for i in range(len(bad_start)-1):
        if bad_start[i+1]>bad_end[i]:
            edges_start.append(bad_start[i])
            edges_end.append(bad_end[i])
    return edges_start, edges_end

def range_plot_regions(chain1, pars1):
    burn1 = int(0.25 * chain1.shape[0])
    index2 = pars1.index('e0')
    f_chain = chain1[burn1:,index2]
    bad_chunks = range_bad_indices(f_chain)
    edges_start, edges_end = range_edges(bad_chunks)
    plt.plot(f_chain)
    for i in range(0,len(edges_start)):
        plt.axvspan(edges_start[i], edges_end[i], color = 'C1', alpha = 0.5)

		
def range_good_indices(f_chain):
    range_tot = max(f_chain)-min(f_chain)

    good_chunks = []
    for i in range(0,len(f_chain)-200):
        range_chunk = max(f_chain[i:i+200])-min(f_chain[i:i+200])
        if np.abs(range_chunk) > 0.15*range_tot:
            good_chunks.append(i)

    for i in range(len(good_chunks)-1):
        if good_chunks[i+1]-good_chunks[1]>1:
            for j in range(200):
                good_chunks.append(good_chunks[i]+j)
                
    return good_chunks

def range_good(chain1, pars1):
    burn1 = int(0.25 * chain1.shape[0])
    index2 = pars1.index('e0')
    f_chain = chain1[burn1:,index2]
    bad_chunks = range_bad_indices(f_chain)
    burn_chain = chain1[burn1:]
    good = np.delete(burn_chain,bad_chunks, axis = 0)
    return good


def mc_range_bad_indices(m_chain):
    range_tot = max(m_chain)-min(m_chain)
    bad_chunks = []
    for i in range(0,len(m_chain)-200):
        range_chunk = max(m_chain[i:i+200])-min(m_chain[i:i+200])
        if np.abs(range_chunk) < 0.15 * range_tot:
            bad_chunks.append(i)

    for i in range(len(bad_chunks)-1):
        if bad_chunks[i+1]-bad_chunks[1]>1:
            for j in range(200):
                bad_chunks.append(bad_chunks[i]+j)
                
    return bad_chunks

def mc_range_edges(bad_chunks):
    bad_start = []
    bad_end = []
    L = bad_chunks
    for k, g in itertools.groupby( enumerate(L), lambda x: x[1]-x[0] ) :
        a = list(map(itemgetter(1), g))
        bad_start.append(a[0])
        bad_end.append(a[-1]+200)

    edges_start = []
    edges_end = []
    for i in range(len(bad_start)-1):
        if bad_start[i+1]>bad_end[i]:
            edges_start.append(bad_start[i])
            edges_end.append(bad_end[i])
    return edges_start, edges_end

def mc_range_plot_regions(chain1, pars1):
    burn1 = int(0.25 * chain1.shape[0])
    index1 = pars1.index('log10_Mc')
    m_chain = chain1[burn1:,index1]
    bad_chunks = mc_range_bad_indices(m_chain)
    edges_start, edges_end = mc_range_edges(bad_chunks)
    plt.plot(m_chain)
    for i in range(0,len(edges_start)):
        plt.axvspan(edges_start[i], edges_end[i], color = 'C1', alpha = 0.5)
		
def mc_range_good_indices(m_chain):
    range_tot = max(m_chain)-min(m_chain)
    good_chunks = []
    for i in range(0,len(m_chain)-200):
        range_chunk = max(m_chain[i:i+200])-min(m_chain[i:i+200])
        if np.abs(range_chunk) > 0.15 * range_tot:
            good_chunks.append(i)

    for i in range(len(good_chunks)-1):
        if good_chunks[i+1]-good_chunks[1]>1:
            for j in range(200):
                good_chunks.append(good_chunks[i]+j)
                
    return good_chunks

def mc_range_good(chain1, pars1):
    burn1 = int(0.25 * chain1.shape[0])
    index1 = pars1.index('log10_Mc')
    m_chain = chain1[burn1:,index1]
    bad_chunks = mc_range_bad_indices(m_chain)
    burn_chain = chain1[burn1:]
    good = np.delete(burn_chain,bad_chunks, axis = 0)
    return good