#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:55:00 2021

@author: jackie_yuan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from obspy.signal.filter import bandpass

def plot_tft(specgram, time, freq, clip=[0., 1.0], log=True, cmap='seismic', zorder=None, title=None):
    'plot spectrogram'
    _range = float(specgram.max() - specgram.min())
    vmin, vmax = clip
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)
    
    kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
              if v is not None}
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if log:
        ax.set_yscale('log')
        
    ax.pcolormesh(time, freq, specgram, norm=norm, **kwargs)
    
    ax.axis('tight')
    ax.set_xlim(0, time[-1]+np.diff(time)[0])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    if title:
        ax.set_title(title)
    ax.grid(False)
    
    plt.show()
    
def _bandpass(data, freqmin, freqmax):
    '''bandpass filter'''
    try:
        nt, nc = data.shape[0], data.shape[1]
        
        datfilt = np.zeros([nt, nc])
        for i in range(nc):
            datfilt[:,i] = bandpass(data[:,i], freqmin, freqmax, 100)
        
    except:
        datfilt = bandpass(data, freqmin, freqmax, 100)
    
    return datfilt 