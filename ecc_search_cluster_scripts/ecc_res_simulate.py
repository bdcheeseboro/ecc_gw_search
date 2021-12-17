#!/usr/bin/env python
# coding: utf-8

# ## Signal Simulation using ecc_res

# #### This notebook is based on the `ecc_res_libstempo.ipynb` by Lankeswar Dey, lanky441@gmail.com. Current version 05/19/2021

# This notebook creates a simulated eccentric gravitational wave dataset using `ecc_res` and `libstempo`



from __future__ import print_function
import sys
import glob
import os
import errno
import pickle
import numpy as np
import libstempo as T
import libstempo.plot as LP, libstempo.toasim as LT
import ephem
import ecc_res
import json
import scipy.constants as sc
import subprocess
import matplotlib.pyplot as plt
import argparse
#from astropy.coordinates import SkyCoord

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove


print(T.__version__)
print(T.libstempo.tempo2version())

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

def call(cmd):
    subprocess.call(cmd,shell=True)
def fit_cmd(parpath,timpath,newpar=True):
    if newpar:
        newpar = '-newpar'        
    cmd = 'tempo2 {0} -f {1} {2}'.format(newpar,parpath,timpath)
    call(cmd)
def fit_psr_t2(psr,partim_dir):
    parpath = partim_dir + psr + '.par'
    timpath = partim_dir + psr + '.tim'
    fit_cmd(parpath,timpath)
    newpath = './new.par'
    fit_cmd(newpath,timpath)
    fit_cmd(newpath,timpath)
    call('mv {0} {1}/{2}.par'.format(newpath, partim_dir,psr))


#####
##  ARGUMENT PARSER
#####
parser = argparse.ArgumentParser(description='injects an eccentric GW into a pta dataset')
    
parser.add_argument('-d', '--datadir',
                    action='store', required=False, default='.',
                    help='directory containing the dataset')
parser.add_argument('-n', '--noisedir',
                    action='store', required=False, default='.',
                    help='directory containing the noise file(s)')
parser.add_argument('-o', '--outdir',
                    action='store', required=False, default='.',
                    help='output directory')
parser.add_argument('--psrs_11',
                    action='store_true', required=False, default=False,
                    help='use 11y psrs list')
args = parser.parse_args()

#Specify paths for various directories needed for making the sim files
datadir = args.datadir
noisepath = args.noisedir
distpath = '/scratch/bdc0001/'

#Designate an output directory for the simulated data files
outdir = args.outdir

#creates the output directory if it does not already exist
if not os.path.exists(os.path.dirname(outdir)):
    try:
        os.makedirs(os.path.dirname(outdir))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

#Grab all parfiles and timfiles
#parfiles = sorted(glob.glob(datadir + 'alternate/NoRedNoisePars/par/*.par'))
psrs_11 = args.psrs_11

if psrs_11:
    parfiles = sorted(glob.glob(datadir + '/*.par'))
    timfiles = sorted(glob.glob(datadir + '/*.tim'))
else:
    parfiles = sorted(glob.glob(datadir + 'par/*.par'))
    timfiles = sorted(glob.glob(datadir + 'tim/*.tim'))

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
        

if psrs_11:
    PsrList_nfix = ['J1909-3744','J2317+1439','J1600-3053','J1640+2224',
                'J1614-2230','J1918-0642','J2043+1711','J0030+0451','J1744-1134',
                'J1741+1351','J0613-0200','B1855+09','J2010-1323','J1455-3330',
                'B1937+21','J0645+5158','J1012+5307','J2145-0750','J1024-0719',
                'J1738+0333','J1910+1256','J1923+2515','J1853+1303','J1944+0907',
                'J2017+0603','J1643-1224','J0340+4130','B1953+29','J2214+3000',
                'J1903+0327','J1747-4036','J2302+4442','J0023+0923']
else:
    PsrList_nfix = ['B1855+09','B1937+21','B1953+29','J0023+0923','J0030+0451',
                'J0340+4130','J0613-0200','J0636+5128','J0645+5158','J0740+6620',
                'J0931-1902','J1012+5307','J1024-0719','J1125+7819','J1453+1902',
                'J1455-3330','J1600-3053','J1614-2230','J1640+2224','J1643-1224',
                'J1713+0747','J1738+0333','J1741+1351','J1744-1134','J1747-4036',
                'J1832-0836','J1853+1303','J1903+0327','J1909-3744','J1910+1256',
                'J1911+1347','J1918-0642','J1923+2515','J1944+0907','J2010-1323',
                'J2017+0603','J2033+1734','J2043+1711','J2145-0750','J2214+3000',
                'J2229+2643','J2234+0611','J2234+0944','J2302+4442','J2317+1439']

#Create a list of tempo pulsar objects
psrs = []
print(PsrList_nfix)
for par, tim in zip(parfiles, timfiles):
    name = par.split('/')[-1].split('_')[0]
    if name in PsrList_nfix:
        psr = T.tempopulsar(parfile = par, timfile = tim, maxobs=100000)
        psrs.append(psr)

with open(distpath+'pulsar_distances.json', 'r') as pdist_file:
    pdist_dict = json.load(pdist_file)

#Now parse this large dictionary so that we can call noise parameters as noise_dict[pulsar name][noise type]
#Returns either floats or 2 column arrays of flags and values.
#Only pulling efac values for right now.
#noise_dict = {}

noise_dict = {}
for psr in psrs:
    noise_dict[psr.name]={}
    noise_dict[psr.name]['equads'] = []
    noise_dict[psr.name]['efacs'] = []
    noise_dict[psr.name]['ecorrs'] = []
    for ky in list(noise_params.keys()):
        if psr.name in ky:
            if 'equad' in ky:
                noise_dict[psr.name]['equads'].append([ky.replace(psr.name + '_' , ''), noise_params[ky]])
            if 'efac' in ky:
                noise_dict[psr.name]['efacs'].append([ky.replace(psr.name + '_' , ''), noise_params[ky]])
            if 'ecorr' in ky:
                noise_dict[psr.name]['ecorrs'].append([ky.replace(psr.name + '_' , ''), noise_params[ky]])
            if 'gamma' in ky:
                noise_dict[psr.name]['RN_gamma'] = noise_params[ky]
            if 'log10_A' in ky:
                noise_dict[psr.name]['RN_Amp'] = 10**noise_params[ky]
                
    noise_dict[psr.name]['equads'] = np.array(noise_dict[psr.name]['equads'])
    noise_dict[psr.name]['efacs'] = np.array(noise_dict[psr.name]['efacs'])
    noise_dict[psr.name]['ecorrs'] = np.array(noise_dict[psr.name]['ecorrs'])    
    
    if len(noise_dict[psr.name]['ecorrs'])==0: #Easier to just delete these dictionary items if no ECORR values. 
        noise_dict[psr.name].__delitem__('ecorrs')    

#c = SkyCoord('02h23m11.4112', '+42d59m31.385s')
'''
#signal injection parameters
#gw postion
gwphi = 0.62478505352
gwtheta = 0.75035284894
log10_dist = 7.929418925714293

#orbital parameters
q = 0.5833333333333334

log10_mc = 9.21748394421
#P0 = 1.05*86400*365.25
log10_forb = -6.713761323432285
e0 = 10**-3
l0 = 0
gamma0 = 0
inc = 1.5707963267948966
psi = 0


#signal injection parameters
#gw postion
gwphi = 5.01
gwtheta = 1.91
log10_dist = 7.5

#orbital parameters
q = 0.6

log10_mc = 9.5
#P0 = 1.05*86400*365.25
log10_forb = -8.5
e0 = 0.1
l0 = 0
gamma0 = 0
inc = np.pi/3
psi = 0
'''
#signal injection parameters
#gw postion
gwphi = 0.62478505352
gwtheta = 0.75035284894
log10_dist = 7.929418925714293

#orbital parameters
q = 0.6

log10_mc = 8.5
#P0 = 1.05*86400*365.25
log10_forb = -6.713761323432285
e0 = 0.1
l0 = 0
gamma0 = 0
inc = np.pi/3
psi = 0

seed_efac = 1234
seed_equad = 5678
seed_jitter = 9101
seed_red = 1121


tmin = [p.toas().min() for p in psrs]
tmax = [p.toas().max() for p in psrs]
tref = max(tmax)

for i, psr in enumerate(psrs):

    #make the pulsar residuals fla
    #LT.make_ideal(psr)
    psr.stoas[:] -= psr.residuals() / 86400.0
    # add efacs
    if len(noise_dict[psr.name]['efacs']) > 0:    
        LT.add_efac(psr, efac = noise_dict[psr.name]['efacs'][:,1], 
                    flagid = 'f', flags = noise_dict[psr.name]['efacs'][:,0], 
                    seed = seed_efac + np.random.randint(len(psrs)))

        ## add equads
        LT.add_equad(psr, equad = noise_dict[psr.name]['equads'][:,1], 
                     flagid = 'f', flags = noise_dict[psr.name]['equads'][:,0], 
                     seed = seed_equad + np.random.randint(len(psrs)))

        ## add jitter
        try: #Only NANOGrav Pulsars have ECORR
            LT.add_jitter(psr, ecorr = noise_dict[psr.name]['ecorrs'][:,1], 
                          flagid='f', flags = noise_dict[psr.name]['ecorrs'][:,0], 
                          coarsegrain = 1.0/86400.0, seed=seed_jitter + np.random.randint(len(psrs)))
        except KeyError:
            pass

        ## add red noise
        LT.add_rednoise(psr, noise_dict[psr.name]['RN_Amp'], noise_dict[psr.name]['RN_gamma'], 
                        components = 30, seed = seed_red + np.random.randint(len(psrs)))
    
    #convert pulsar sky location to proper frame
    fac = 180./np.pi
    coords = ephem.Equatorial(ephem.Ecliptic(str(psr['ELONG'].val*fac), 
                                                 str(psr['ELAT'].val*fac)))
    ptheta = np.pi/2 - float(repr(coords.dec))
    pphi = float(repr(coords.ra))
    #Inject signal into set of pulsars
    
    toas = psr.toas()*86400 #toas
    if psr in pdist_dict.keys():
         pdist = pdist_dict[psr][0] #distance of pulsar in kpc
    else:
        pdist = 1

    residuals = ecc_res.add_ecc_cgw(toas, ptheta, pphi, pdist, gwtheta, gwphi, log10_mc, q, log10_forb, e0, l0, gamma0, 
                    inc, psi, log10_dist, pphase = None, gamma_P = None, tref = tref, #tref must be greater than the last TOA MJd value
                    psrterm = True, evol = True, waveform_cal = True, res = 'Both')
    psr.stoas[:] += (residuals)/86400 #converting to days
    
    psr.savepar(outdir + psr.name + '_simulate.par') #saves the simulated par file w/ ecc signal
    psr.savetim(outdir + psr.name + '_simulate.tim') #saves the simulated tim file w/ ecc signal

for psr in psrs:
    fd, abs_path = mkstemp()
    with fdopen(fd,'w') as new_file:
        with open(outdir+psr.name+'_simulate.par','r') as old_file:
            # read content from first file
            for line in old_file:
                line_split = line.split(' ')
                if 'DMX_' in line_split[0]:
                    if len(line_split) == 15:
                        line_split[9] = str(0)
                        new_line = " ".join(line_split)
                        new_file.write(new_line)
                    else:
                        line_split[8] = str(0)
                        new_line = " ".join(line_split)
                        new_file.write(new_line)
                else:
                    new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(outdir+psr.name+'_simulate.par', abs_path)

    #Remove original file
    remove(outdir+psr.name+'_simulate.par')

    #Move new file
    move(abs_path, outdir+psr.name+'_simulate.par')
    
    #refit the files for the injected signal
    fit_psr_t2(psr.name+'_simulate' ,outdir)


inj_params = {'gwphi': gwphi, 'gwtheta': gwtheta, 'log10_dist': log10_dist, 'q': q, 'log10_mc': log10_mc, 'log10_forb': log10_forb, 'e0': e0, 'l0': l0, 'gamma0': gamma0, 'inc': inc, 'psi': psi}

#save injection parameters as a dictionary
ecc_dict_dump = json.dumps(inj_params)
f = open(outdir+"ecc_inj_params.json","w")
f.write(ecc_dict_dump)
f.close()

