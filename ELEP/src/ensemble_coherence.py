import scipy.ndimage as nd
import numpy as np

def neighbor_sum(arr):
    return sum(arr)

def ensemble_semblance(signals, paras):
    '''
    Function: calculate coherence or continuity via semblance analysis.
    Reference: Marfurt et al. 1998
    
    PARAMETERS:
    ---------------
    signals: input data [ntraces, npts]
    paras: a dict contains various parameters used for calculating semblance. 
           Note: see details in Tutorials.
    
    RETURNS:
    ---------------
    semblance: derived cohence vector [npts,]
    
    Written by Congcong Yuan (Jan 05, 2023)
    
    '''
    # setup parameters
    ntr, npts = signals.shape 
    dt = paras['dt']
    semblance_order = paras['semblance_order']
    semblance_win = paras['semblance_win']
    weight_flag = paras['weight_flag']
    window_flag = paras['window_flag']
    
    semblance_nsmps = int(semblance_win/dt)
    
    # initializing
    semblance, v = np.zeros(npts), np.zeros(npts)
    
    # sums over traces
    square_sums = np.sum(signals, axis=0)**2
    sum_squares = np.sum(signals**2, axis=0)
    
    # loop over all time points
    if weight_flag:
        if weight_flag == 'max':
            v = np.amax(signals, axis=0)
        elif weight_flag == 'mean':
            v = np.mean(signals, axis=0)
        elif weight_flag == 'mean_std':
            v_mean = np.mean(signals, axis=0)
            v_std = np.mean(signals, axis=0)
            v = v_mean/v_std
    else:
        v = 1.
        
    if window_flag:
        # sum over time window
        sums_num = nd.generic_filter(square_sums, neighbor_sum, semblance_nsmps, mode='constant')
        sums_den = ntr*nd.generic_filter(sum_squares, neighbor_sum, semblance_nsmps, mode='constant')
    else:
        sums_num = square_sums
        sums_den = ntr*sum_squares

    # original semblance
    semblance0 = sums_num/sums_den

    # enhanced semblance
    semblance = semblance0**semblance_order*v 
    
    return semblance  