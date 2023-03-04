def ensemble_statistics(signals, paras):
    '''
    Simple methods for ensemble estimation.
    
    PARAMETERS
    -----------
    signals: np.array[ntraces, npts]
    methods: max or mean
    '''
    dt = paras['dt']
    win = paras['win']
    window_flag = paras['window_flag']
    method = paras['method']

    ntr, npts = signals.shape 
    nwd = np.floor(npts/(win/dt))
    
    output = np.zeros([npts,])
    if window_flag:
        for iwd in range(int(nwd)-1):
            twind = np.arange(iwd*int(win/dt), (iwd+1)*int(win/dt))
            if method == 'max':
                ind = np.argwhere(signals[:,twind]==np.amax(signals[:,twind]))[0,0]
                output[twind] = signals[ind,twind]
            elif method == 'mean':
                output[twind] = np.mean(signals[:,twind], axis=0)
            elif method == 'weight_mean':
                weights = np.max(signals[:,twind], axis=0)
                subsigs = np.array([signals[itr,twind]*weights[itr] for itr in range(ntr)])
                output[twind] = np.mean(subsigs, axis=0)
            else:
                raise ValueError('No such method is defined!')
            
        # finish the last part
        if twind[1]<npts:
            twind = np.arange(twind[1], npts-1)
            if method == 'max':
                ind = np.argwhere(signals[:,twind]==np.amax(signals[:,twind]))[0,0]
                output[twind] = signals[ind,twind]
            elif method == 'mean':
                output[twind] = np.mean(signals[:,twind], axis=0)
            elif method == 'weight_mean':
                weights = np.amax(signals[:,twind], axis=0)
                subsigs = np.array([signals[itr,twind]*weights[itr] for itr in range(ntr)])
                output[twind] = np.mean(subsigs, axis=0)
            else:
                raise ValueError('No such method is defined!')
            
    else:
        if method == 'max':
            ind = np.argwhere(signals[:,:]==np.amax(signals[:,:]))[0,0]
            output = signals[ind,]
        elif method == 'mean':
            output = np.mean(signals[:,:], axis=0)
        elif method == 'weight_mean':
            weights = np.amax(signals[:,:], axis=1)
            subsigs = np.array([signals[itr,:]*weights[itr] for itr in range(ntr)])
            output = np.mean(subsigs, axis=0)
        else:
            raise ValueError('No such method is defined!')
            
    return output 