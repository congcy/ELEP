#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:55:00 2021

@author: jackie_yuan
"""

import numpy as np
from obspy import read

# ------------------------- filtering ----------------------
def make_LinFq(f_min, f_max, delta, nfreq):
    """Calculate linearly spaced frequency array for MB filtering."""
    f_ny = float(1./(2*delta))
    if f_max > f_ny:
        f_max = f_ny
    freq = np.linspace(f_min, f_max, nfreq)
    return freq


def make_LogFq(f_min, f_max, delta, nfreq):
    """Calculate log spaced frequency array for MB filtering."""
    f_ny = float(1./(2*delta))
    if f_max > f_ny:
        f_max = f_ny
    freq = np.logspace(np.log2(f_min), np.log2(f_max), nfreq, base=2)
    return freq

# ------------------------- coefficients ----------------------
def rec_filter_coeff(freqs, delta):
    freqs = np.array(freqs, ndmin=1)
    nyq = 1. / (2*delta)
    T = 1. / freqs
    w = T / (2*np.pi)
    rel_freqs = freqs / nyq
    # empirical approach to fix filter shape close to the Nyquist:
    w[rel_freqs >= 0.2] /= (rel_freqs[rel_freqs >= 0.2] * 7)
    C_HP = w / (w + delta)      # high-pass filter constant
    C_LP = delta / (w + delta)  # low-pass filter constant
    return C_HP, C_LP

def _gausscoeff(sigma, A, B):
    '''implementation of the algorithm by Young&Vliet, 1995'''
    if sigma > 0.5:
        q = 0.98711*sigma - 0.96330
    elif sigma == 0.5:
        q = 3.97156 - 4.14554 * np.sqrt(1.0 - 0.26891*sigma)
    else:
        raise ValueError("Sigma for Gaussian filter must be >=0.5 samples.\n")

    b = np.zeros(4)
    b[0] = 1.57825 + 2.44413*q + 1.4281*pow(q, 2) + 0.422205*pow(q, 3)
    b[1] = 2.44413*q + 2.85619*pow(q, 2) + 1.26661*pow(q, 3)
    b[2] = -(1.4281*pow(q, 2) + 1.26661*pow(q, 3))
    b[3] = 0.422205*pow(q, 3)

    B[0] = 1.0 - ((b[1] + b[2] + b[3])/b[0])

    A[0] = 1
    for i in range(1,4):
        A[i] = -b[i]/b[0]   
    nA, nB = 4, 1
    
    return A, nA, B, nB

# ----------------------------- SNR --------------------------
def get_noisy_trace(sig, noi, itp, its, snr, dit=300):
    ns, ns2 = len(sig), len(noi)
    assert ns == ns2 
    if ns-its > dit:
        sdata = np.percentile(sig[its:its+dit],95)
    else:
        sdata = np.percentile(sig[its:ns],95)    
    if itp > dit:
        ndata = np.percentile(noi[itp-dit:itp],95)
    else:
        ndata = np.percentile(noi[0:itp],95)   
    fact = sdata/np.sqrt(10**(snr/10))
    
    return sig+fact/ndata*noi

def get_snr_pc95(data, itp, its, dit=300):
    ns = len(data)
    if ns-its > dit:
        sdata = np.percentile(data[its:its+dit],95)**2
    else:
        sdata = np.percentile(data[its:ns],95)**2
    if itp > dit:
        ndata = np.percentile(data[itp-dit:itp],95)**2
    else:
        ndata = np.percentile(data[0:itp],95)**2
    if (ndata==0) or (sdata==0):
        return 0.
    else:
        return 10*np.log10(sdata/ndata)

# -----------------------------------------------------------
def create_obspy_trace(trace_dict):
    # initiate
    trace = read()

    # write to data dict
    for ic in range(trace_dict['n_channels']):
        # write in header
        trace[ic].stats.network = trace_dict['network']
        trace[ic].stats.station = trace_dict['station']
        if trace_dict['location']:
            trace[ic].stats.location = trace_dict['location']
            if trace_dict['frequency_band']:
                ifb = trace_dict['frequency_band_index']
                fb = trace_dict['frequency_band']
                trace[ic].stats.location = trace_dict['location']+'_fb'+str('%02d'%ifb)
        else:
            if trace_dict['frequency_band']:
                ifb = trace_dict['frequency_band_index']
                fb = trace_dict['frequency_band']
                trace[ic].stats.location = 'fb'+str('%02d'%ifb) #+str('%.2f'%fb[0])+'_'+str('%.2f'%fb[1])+'Hz'
        trace[ic].stats.channel = trace_dict['channel']+trace_dict['components'][ic]
        trace[ic].stats.starttime = trace_dict['starttime']
        # trace[ic].stats.endtime = trace_dict['endtime']
        trace[ic].stats.sampling_rate = trace_dict['sampling_rate']
        trace[ic].stats.delta = trace_dict['delta']
        trace[ic].stats.npts = trace_dict['npts']
        trace[ic].stats.back_azimuth = trace_dict['back_azimuth']
        
        # write in data
        trace[ic].data = trace_dict['data'][:,ic]

    return trace


def normalize(data, mode = 'max'): 
    'Normalize waveforms in a batch'
        
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data 
        return data, max_data   

    elif mode == 'absmax':
        max_data = np.max(np.abs(data), axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data 
        return data, max_data           

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
        return data, std_data 