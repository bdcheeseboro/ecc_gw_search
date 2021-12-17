import numpy as np
from numpy import pi, sin, cos, sqrt
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
import scipy.constants as sc
import matplotlib.pyplot as plt

import ecc_utils as eu


GMsun = 1.327124400e20  # measured more precisely than Msun alone!
c = sc.speed_of_light
TSUN = GMsun / c**3
parsec = sc.parsec
KPC2S = parsec/c * 1e3
PC2S = parsec/c


def get_hA_hB(i, e, u, phi, delta):
	ci = cos(i)
	si = sin(i)
	chi = e * cos(u)
	xi = e* sin(u)
	OTS = sqrt(1 - e*e)
	
	h_mq_A = ((-2 * (ci*ci + 1) * OTS * xi * sin(2*phi) + (ci*ci + 1) * (2*e*e
		  - chi*chi + chi -2) * cos(2*phi) + si*si * (1 - chi) * chi)/(1 - chi)**2)
	
	h_mq_B = 2*ci * (2 * OTS * xi * cos(2*phi) + (2*e*e - chi*chi + chi -2) * sin(2*phi))/(1 - chi)**2
	
	h_cq_A = (delta * si * ((ci**2 + 1) * OTS * (6 *chi**2 - 7*chi - 8*e**2 + 9) * cos(3*phi)
				+ 2 * (ci**2 + 1) * xi * (chi**2 - 2*chi - 4*e**2 + 5) * sin(3*phi)
				+ OTS * (1-chi) * ((6*ci**2 - 2)*chi - ci**2 - 5) * cos(phi)
				+ 2 * (1 - 3*ci**2) * (1-chi)**2 * xi * sin(phi))/(4.*(1 - chi)**3))
	h_cq_B = (ci*si*delta * (2*xi * (1 - chi)**2 * cos(phi) + 2*xi * (-5 + 4*e**2 + 2*chi - chi**2)*cos(3*phi) 
				+ OTS * (1 - chi) * (-3 + 2*chi) * sin(phi)	+ OTS * (9 - 8*e**2 - 7*chi + 6*chi**2)*sin(3*phi))/(2.*(1 - chi)**3))
	
	return(h_mq_A, h_mq_B, h_cq_A, h_cq_B)


def calculate_sp_sx(toas, gwdist, mc, q, n0, e0, l0, gamma0, inc, psi, 
				tref, Fp, Fx, evol, waveform_cal):
    
    toa_sample = np.linspace(np.min(toas), np.max(toas), int((np.max(toas) - np.min(toas))/86400), endpoint=True)
	
    #m = (((1+q)**2)/q)**(3/5) * mc
    eta = q/(1+q)**2
    m = mc/eta**(3/5)
    delta = (1-q)/(1+q)
    e02 = e0*e0

    if evol:
        ns, es, ls, gammas = eu.evolve_orbit(toa_sample, mc, q, n0, e0, l0, gamma0, tref)
    else:
        ns = np.full(len(toa_sample), n0)
        es = np.full(len(toa_sample), e0)
        
        k0 = eu.get_k(n0, e0, m, eta)
        ls = l0 + n0 * (toa_sample - tref)
        gammas = gamma0 + k0 *n0 * (toa_sample - tref)
        
    
    emax = np.max(es)
    if emax >= 1.0:
        #print("e_max =", emax)
        #print("Encounter e >= 1.0! Setting return to NaN!!")
        #print('checking emax')
        return np.full(len(toas), np.nan)

        
    xs, ks, ephis = eu.get_x_k_ephi(ns, es, m, eta)
    
    ephimax = np.max(ephis)
    if ephimax >= 1.0:
        #print("ephi_max =", ephimax)
        #print("Encounter ephi >= 1.0! Setting return to NaN!!")
        #print('checking ephimax')
        return np.full(len(toas), np.nan)

    kmax = np.max(ks)    
    if kmax >= 0.5:
        #print("k_max =", kmax)
        #print("Encounter k_max >= 1.0! Binary is out of inspiral phase! Setting return to NaN!!")
        #print('checking kmax')
        return np.full(len(toas), np.nan)

    
    if waveform_cal:
        us, phis = eu.get_u_phi(xs, ks, ephis, es, ls, gammas)
                
        H0 = TSUN * m * eta * xs / (gwdist * PC2S)
        h_mq_A, h_mq_B, h_cq_A, h_cq_B = get_hA_hB(inc, es, us, phis, delta)
    
        hA = h_mq_A + sqrt(xs) * h_cq_A
        hB = h_mq_B + sqrt(xs) * h_cq_B
    
        hps = H0 * (hA * cos(2*psi) - hB * sin(2*psi))
        hxs = H0 * (hB * cos(2*psi) + hA * sin(2*psi))
        hs = Fp * hps + Fx * hxs
    
        res = cumtrapz(hs, x = toa_sample, initial=0)
        res_fun = interp1d(toa_sample, res)
        residuals = res_fun(toas)
        
        return (residuals)
