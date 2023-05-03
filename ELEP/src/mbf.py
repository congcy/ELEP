#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:57:47 2021

refer to https://github.com/BackTrackBB/backtrackbb/blob/master/backtrackbb

@author: jackie_yuan
"""
import numpy as np
import scipy.ndimage as nd
from mbf_utils import _gausscoeff
import time
from obspy.signal.invsim import cosine_taper

# --------------------- filtering ------------------------
def _recursive_filter(signal, C_HP, C_LP, npoles=2):
    '''Bandpass (or highpass) filtering by cascade of simple, first-order recursive highpass and lowpass filters. 
    The number of poles gives the number of filter stages.'''
    
    # set up parameters
    _prev_sample_value = 0.
    npts = len(signal)
    
    # initializing 
    _filterH = np.zeros(npts)
    _filterH0 = np.zeros(npts) 
    _filterL = np.zeros(npts)
    
    filt_signal = np.zeros(npts)
    
    # loop time samples
    for i in range(npts):
        # broadband filter = npoles single-pole filter
        for n in range(npoles):
            if n == 0:
                s0 = _prev_sample_value
                s1 = signal[i]
            else:
                s0 = _filterH0[n-1]
                s1 = _filterH[n-1]
            _filterH0[n] = _filterH[n]
            _filterH[n] = C_HP * (_filterH[n] + s1 - s0)
            
        if C_LP < 0:
            # high-pass filter
            filt_signal[i] = _filterH[npoles-1]
        else:
            for n in range(npoles):
                if n == 0:
                    s0 = _filterH[npoles-1]
                else:
                    s0 = _filterL[n-1]
                _filterL[n] = _filterL[n] + C_LP * (s0 - _filterL[n])
                
            filt_signal[i] = _filterL[npoles-1]
            
        _prev_sample_value = signal[i]
        
    return filt_signal

def MB_filter(signal, paras):
    '''multiple bandpass filtering by using _recursive_filtering() function:
    Performs MBfiltering using 2HP+2LP recursive filter'''
    
    # set up parameters
    frequencies = paras['frequencies'] 
    dt = paras['dt']
    CN_HP, CN_LP = paras['CN_HP'], paras['CN_LP']
    npoles = paras['npoles']
    
    Nb = len(frequencies)
    npts = len(signal)
    
    # initializing
    FN = np.zeros((Nb, npts), float)

    # normalization coefficients
    filter_norm = rec_filter_norm(frequencies, dt, CN_HP, CN_LP, npoles)
    
    # loop over frequency bands
    for n in range(Nb):
        FN[n] = _recursive_filter(signal, CN_HP[n], CN_LP[n], npoles=2)
        FN[n] /= filter_norm[n]
    return FN

def _lfilter(signal, A, nA, B, nB):
    npts = len(signal)
    filt_signal = np.zeros(npts)
    for n in range(npts):
        filt_signal[n] = 0
        for nb in range(nB):
            if nb > n: break
            filt_signal[n] += B[nb] * signal[n-nb]
        for na in range(1, nA):
            if na > n: break
            filt_signal[n] -= A[na] * filt_signal[n-na]
        filt_signal[n] /= A[0]
    
    return filt_signal 
    
def _reverse(signal):
    npts = len(signal)
    rev_signal = np.copy(signal)

    end = npts - 1
    for n in range(npts//2):
        tmp = rev_signal[n]
        rev_signal[n] = rev_signal[end]
        rev_signal[end] = tmp
        end -=1
        
    return rev_signal
    
def recursive_gauss_filter(signal, sigma):
    npts = len(signal)
    
    A, B = np.zeros(4), np.zeros(4)
    A, nA, B, nB = _gausscoeff(sigma, A, B)

    rev_filt_signal = np.zeros(npts)
    filt_signal = np.zeros(npts)

    filt_signal = _lfilter(signal, A, nA, B, nB)
    rev_filt_signal = _reverse(filt_signal)
    filt_signal = _lfilter(rev_filt_signal, A, nA, B, nB)
    rev_filt_signal = _reverse(filt_signal)
    
    return rev_filt_signal

def GaussConv(data_in, sigma):
    derivative_data = np.gradient(data_in)
    derivative_data[derivative_data < 0] = 0
    CF_gaussian = recursive_gauss_filter(derivative_data, sigma)
    
    return CF_gaussian

def CF_summary(signals, paras):
    ''''stack 2D filtered signals to 1D signal'''
    summary_type = paras['CF_summary_type']    
    if summary_type == 'envelope':
        rms_argmax = np.amax(signals, axis=0)
        signals_sum = np.sqrt(np.power(signals, 2).mean(axis=0)) 
        return rms_argmax, signals_sum 
    
    elif summary_type == 'kurtosis':
        
        if paras['sigma_gauss']:
            sigma_gauss = int(paras['sigma_gauss']/paras['dt'])
        else:
            sigma_gauss = int(paras['CF_decay_win']/paras['dt']/2)
        
        kurt_argmax = np.amax(signals, axis=0)
        signals_sum = GaussConv(kurt_argmax, sigma_gauss)

        return kurt_argmax, signals_sum
        
# ------------------------- normalization ----------------------
def rec_filter_norm(freqs, dt, CN_HP, CN_LP, npoles):
    """Empirical approach for computing filter normalization coefficients."""
    freqs = np.array(freqs, ndmin=1)
    norm = np.zeros(len(freqs))
    for n, freq in enumerate(freqs):
        length = 4. / freq
        time = np.arange(0, length+dt, dt)
        signal = np.sin(freq * 2 * np.pi * time)
        signal_filt = _recursive_filter(signal, CN_HP[n], CN_LP[n], npoles)
        norm[n] = signal_filt.max()
    return norm 

# --------------------------- characteristic features (CF) -----------------------
def _recursive_hos(signal, C_WIN, order, sigma_min):
    '''apply recursive high-order statistics, for impulsive signals'''

    npts, n_win = len(signal), int(1/C_WIN)
    power = order//2
    _mean, _var = np.mean(signal[:n_win]), np.std(signal[:n_win])
    hos_signal = np.zeros(signal.shape)
    
    # initialize
    for i in range(n_win):
        _mean = C_WIN * signal[i] + (1 - C_WIN) * _mean;
        _var = C_WIN * pow((signal[i] - _mean), 2.0) + (1 - C_WIN) * _var;

    _hos=0.
    for i in range(npts):
        _mean = C_WIN * signal[i] + (1 - C_WIN) * _mean
        var_temp = C_WIN * pow((signal[i] - _mean), 2.0) + (1 - C_WIN) * _var
        if var_temp > sigma_min:
            # if sigma_min < 0, this is always true 
            _var = var_temp
        else:
            _var = sigma_min
        
        _hos = C_WIN * (pow((signal[i] - _mean), order) / pow(_var, power)) + (1 - C_WIN) * _hos
        hos_signal[i] = _hos
        
    return _mean, _var, hos_signal
    
def _recursive_rms(signal, C_WIN):
    '''apply energy envelope for emergent signals'''
    npts, n_win = len(signal), int(1/C_WIN)
    rms_signal = np.zeros(signal.shape)
    
    _mean_sq=0.
    # initialize
    for j in range(n_win):
        _mean_sq = _mean_sq + pow(signal[j], 2)
    _mean_sq = np.sqrt(_mean_sq/n_win)
    
    for i in range(npts):
        _mean_sq = np.sqrt(C_WIN * pow(signal[i], 2.) + (1 - C_WIN) * pow(_mean_sq, 2.))
        rms_signal[i] = _mean_sq
    
    return _mean_sq, rms_signal

def computer_CF(signal_fn, paras):
    ''' calculates the characteristic function (CF)
    for each band. ''' 
    frequencies = paras['frequencies']    
    dt = paras['dt']
    var_w = paras['CF_var_w']
    CF_type = paras['CF_type']
    CF_decay_win = paras['CF_decay_win']
    hos_order = paras['hos_order']
    hos_sigma = paras['hos_sigma']
    
    Tn = 1. / frequencies
    CF_decay_nsmps = CF_decay_win / dt
    Nb = len(frequencies)
    
    if hos_sigma is None:
        hos_sigma = -1.
        
    # initializing
    CF = np.zeros(signal_fn.shape, float)
    
    # loop over freq bands
    for n in range(Nb):
        # Define the decay constant
        if var_w and CF_type == 'envelope':
            # CF_decay_nsmps_mb = (Tn[n]/dt) * CF_decay_nsmps
            CF_decay_nsmps_mb = Tn[n] * CF_decay_nsmps
        else:
            CF_decay_nsmps_mb = CF_decay_nsmps
        
        CF_decay_constant = 1 / CF_decay_nsmps_mb
        # Calculates CF for each MBF signal
        if CF_type == 'envelope':
            _, CF[n] = _recursive_rms(signal_fn[n], CF_decay_constant)
            
        if CF_type == 'kurtosis':
            _, _, CF[n] = _recursive_hos(signal_fn[n], CF_decay_constant, hos_order, hos_sigma)
    
    return CF