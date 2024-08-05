import os
import pynwb
import warnings

import numpy  as np
import pickle as pkl

from itertools   import product
from parallelbar import progress_imap
from pathlib import Path

from dlab.psth_and_raster import trial_by_trial
from tqdm import tqdm

## USER INPUT BELOW FUNCTIONS
# MAKE SURE YOU CHANGE DESIRED INPUTS AND PARAMETERS BELOW
######################################################################################

def setup(x,y):
    global signal, noise
    signal = x
    noise  = y

def pearson_corr(arr1, arr2):
    """
    Computes Pearson correlation along the last dimension of two multidimensional arrays.
    """
    mean1          = np.mean(arr1, axis=-1, keepdims=True)
    mean2          = np.mean(arr2, axis=-1, keepdims=True)

    dev1, dev2     = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer          = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2     = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom          = np.sqrt(var1 * var2)

    # Divide numerator by denominator, but use NaN where the denominator is 0
    return np.divide(numer, denom, out = np.full_like(numer, np.nan), where=(denom != 0))

def average_ranks(arr):
    """
    Computes the ranks of the elements of the given array along the last dimension.
    For ties, the ranks are _averaged_. Returns an array of the same dimension of `arr`. 
    """
    sorted_inds   = np.argsort(arr, axis=-1)  # Sorted indices
    ranks, ranges = np.empty_like(arr), np.empty_like(arr)
    ranges        = np.tile(np.arange(arr.shape[-1]), arr.shape[:-1] + (1,))

    np.put_along_axis(ranks, sorted_inds, ranges, -1)
    ranks      = ranks.astype(int)
    sorted_arr = np.take_along_axis(arr, sorted_inds, axis=-1)
    diffs      = np.diff(sorted_arr, axis=-1)
    
    # Pad with an extra zero at the beginning of every subarray
    pad_diffs = np.pad(diffs, ([(0, 0)] * (diffs.ndim - 1)) + [(1, 0)])
    
    # Wherever the diff is not 0, assign a value of 1; this gives a set of small indices
    # for each set of unique values in the sorted array after taking a cumulative sum
    pad_diffs[pad_diffs != 0] = 1
    unique_inds  = np.cumsum(pad_diffs, axis=-1).astype(int)
    unique_maxes = np.zeros_like(arr)  # Maximum ranks for each unique index
    
    # Using `put_along_axis` will put the _last_ thing seen in `ranges`, which will result
    # in putting the maximum rank in each unique location
    np.put_along_axis(unique_maxes, unique_inds, ranges, -1)
    
    # We can compute the average rank for each bucket (from the maximum rank for each bucket)
    # using some algebraic manipulation
    diff        = np.diff(unique_maxes, prepend=-1, axis=-1)  # Note: prepend -1!
    unique_avgs = unique_maxes - ((diff - 1) / 2)
    avg_ranks   = np.take_along_axis(unique_avgs, np.take_along_axis(unique_inds, ranks, -1), -1)
    
    return avg_ranks
        
def process_pair(args):
    warnings.simplefilter("ignore",category=Warning)
    pair, B= args
    
    if type(B) is tuple:
        signal0 = signal[:,B[0]*pair[0]:B[0]*(pair[0]+1)]
        signal1 = signal[:,B[0]*pair[1]:B[0]*(pair[1]+1)]
        noise0  = noise[:, B[1]*pair[0]:B[1]*(pair[0]+1)]
        noise1  = noise[:, B[1]*pair[1]:B[1]*(pair[1]+1)]
    
    else:
        signal0 = signal[:,B*pair[0]:B*(pair[0]+1)]
        signal1 = signal[:,B*pair[1]:B*(pair[1]+1)]
        noise0  = noise[:, B*pair[0]:B*(pair[0]+1)]
        noise1  = noise[:, B*pair[1]:B*(pair[1]+1)]
        
    # Calculate spearman and pearson correlations for signal using numpy (converted back from GPU tensor)
    sp_signal = np.nanmean(pearson_corr(average_ranks(signal0),average_ranks(signal1)))
    pe_signal = np.nanmean(pearson_corr(signal0, signal1))

    sp_noise = np.nanmean(pearson_corr(average_ranks(noise0),average_ranks(noise1)))
    pe_noise = np.nanmean(pearson_corr(noise0,noise1))

    return np.array([sp_signal,sp_noise,pe_signal,pe_noise])


def calculate_pw_corrs(pairs, B,signal,noise):
    Nchunk = int(len(pairs) / os.cpu_count())
    res = progress_imap(process_pair, [(pair, B) for pair in pairs],
                        n_cpu               = os.cpu_count(),
                        total               = len(pairs),
                        chunk_size          = Nchunk,
                        initializer         = setup,
                        initargs            = (signal,noise),
                        context             ='spawn',
                        # return_failed_tasks = True,
                        )
    
    return np.array(res)
#############################################################################################################

        
if __name__ == '__main__':

    params = {'mouse'      : 'D5',
              'rec_date'   : '2023-11-18',
              'nwb_path'   : r'd:\d5\juan_D5_2023-11-18.nwb',
              'output_dir' : r'd:\\',
            #   'output_dir' : r'C:\Users\juans\github\color_and_form\results',
              'stimulus'   : 'color_exchange',
              'window'     : 0.3, #response window to measure correlations
              'bin_sizes'  : [0.01, 0.05, 0.1],
              'unit_filter': True,
              'mode'       : 'trial_sum'
            }
    
    """
    Modes
    - full          : all bins from signal/noise 
    - trial_sum     : Sum of spike counts for each trial
    - trial_avg     : Signal trial averaged
    - per_condition : Probably only works for color stim for now.
    """
    for key in params.keys():
        print(f'{key}: {params[key]}')
        
    print()
    proceed = input('Would you like to proceed?[y\\n]')
    if proceed != 'y':
        quit()
        
    out_dir    = os.path.join(params['output_dir'],f'{params['mouse'].upper()}','data','signal_noise_corr')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    OUTPUT = {}

    print('Loading Data...')
    print()
    
    io = pynwb.NWBHDF5IO(params['nwb_path'], mode='r')
    nwb = io.read()

    units_df   = nwb.units.to_dataframe()
    if params['unit_filter']:
        units_good = units_df[(units_df.label == 1)|(units_df.label == 3)] #Specific to how Juan labels NWBs for bombcell outputs
        units_df   = None 
        units_good.reset_index(inplace=True,drop=True)
    else:
        units_good = units_df
        units_df   = None
    

    stim_df    = nwb.trials.to_dataframe()
    stim_times = stim_df.loc[stim_df.stimulus == params['stimulus'],'start_time'].values

    if params['mode'] == 'per_condition':
        if params['stimulus'] == 'color_exchange':
            n_seg  = 8
            deltaM = np.diff(stim_df.loc[stim_df.stimulus == params['stimulus'],'green'].values)/2
            deltaS = np.diff(stim_df.loc[stim_df.stimulus == params['stimulus'],'uv'].values)/2
            data   = np.round(np.array((deltaM,deltaS)),5).T
            angles = np.arctan2(data[:, 1], data[:, 0])+np.deg2rad(45)
            
            segment_indices = (np.round(((angles + np.pi) / (2*np.pi)) * n_seg) % n_seg).astype(int)
            
        else:
            print("per_condition only works for color exchange")
        
        
    spike_data   = [np.array(i) for i in units_good.spike_times]
    spike_data   = [i[(i > stim_times[0]) & (i < stim_times[-1]+params['window'])] for i in spike_data]
    n_units      = len(spike_data)
    pairs        = list(product(np.arange(n_units),np.arange(n_units)))
    
    OUTPUT['unit_info'] = units_good[['probe','cluster_id', 'channel', 'xpos', 'ypos', 'depth','KSamplitude','label']].to_dict()
    
    params['n_units'] = n_units
    params['pairs']   = pairs
    OUTPUT['params']  = params
    
    units_good = None
    stim_df    = None
    nwb        = None

    OUTPUT['correlations'] = {}

    for bin_ in params['bin_sizes']:
        print(f'Bin Size  = {bin_} seconds')
        B = np.round(np.ceil(params['window']/bin_)).astype(int)
        N = len(spike_data)
        
        print('Gathering Stimulus-Evoked Activity...')
        
        if params['mode'] == 'per_condition' and params['stimulus'] == 'color_exchange':
            B = (n_seg,np.round(np.ceil(params['window']/bin_)).astype(int))
            
            signal = np.zeros((B[0]*N))
            noise  = np.zeros((len(stim_times)-1,B[1]*N))
            
            for i, times in enumerate(spike_data):
                k  = 0
                tc = []
                for c in range(n_seg):
                    psth,_,_,bytrial = trial_by_trial(times,stim_times[1:][segment_indices == c],0,params['window'],bin_)
                    
                    tc.append(bytrial.mean())
                    noise[k:k+len(bytrial),B[1]*i:B[1]*(i+1)] = bytrial
                    
                    k += len(bytrial)
                    
                signal[B[0]*i:B[0]*(i+1)]  = tc
                
            signal  = signal.reshape(1,-1)
                
                
        
        if params['mode'] == 'full':
        
            tensor = np.zeros((len(stim_times),N*B))
            
            for i, times in tqdm(enumerate(spike_data),total = len(spike_data)):
                _,_,_,bytrial = trial_by_trial(times,stim_times,0,params['window'],bin_)
                
                tensor[:,B*i:B*(i+1)] = bytrial
            
            signal  = tensor.mean(axis=0)
            signal  = signal.reshape(1,-1)
            noise   = tensor - signal
            tensor  = None
            
        if params['mode'] == 'trial_sum':
            B = (len(stim_times),np.round(np.ceil(params['window']/bin_)).astype(int))
            
            signal = np.zeros(N*B[0])
            noise  = np.zeros((len(stim_times),N*B[1]))
            
            for i, times in tqdm(enumerate(spike_data),total = len(spike_data)):
                psth,_,_,bytrial = trial_by_trial(times,stim_times,0,params['window'],bin_)
                
                signal[B[0]*i:B[0]*(i+1)]  = bytrial.sum(axis=1)
                noise[:,B[1]*i:B[1]*(i+1)] = bytrial

            signal = signal.reshape(1,-1)
                
            
        if params['mode'] == 'trial_avg':
            B = (len(stim_times),np.round(np.ceil(params['window']/bin_)).astype(int))
            
            signal = np.zeros(N*B[0])
            noise  = np.zeros((len(stim_times),N*B[1]))
            
            for i, times in tqdm(enumerate(spike_data),total = len(spike_data)):
                psth,_,_,bytrial = trial_by_trial(times,stim_times,0,params['window'],bin_)
                
                signal[B[0]*i:B[0]*(i+1)]  = bytrial.mean(axis=1)
                noise[:,B[1]*i:B[1]*(i+1)] = bytrial

            signal = signal.reshape(1,-1)
                
        print('Calculating Correlations....')
        print('May take a while to get started')

        res = calculate_pw_corrs(pairs,B,signal,noise)

        OUTPUT['correlations'][f'{int(bin_*1000)}ms'] = {}
        OUTPUT['correlations'][f'{int(bin_*1000)}ms']['signal_spearman'] = res[:,0]
        OUTPUT['correlations'][f'{int(bin_*1000)}ms']['noise_spearman']  = res[:,1]
        OUTPUT['correlations'][f'{int(bin_*1000)}ms']['signal_pearson']  = res[:,2]
        OUTPUT['correlations'][f'{int(bin_*1000)}ms']['noise_pearson']   = res[:,3]

    print('Saving Output...')
    with open(os.path.join(out_dir,f'{params['mouse']}_sn_corr_{params['stimulus']}.pkl'),'wb') as f:
        pkl.dump(OUTPUT,f)
        print(f'Output saved to {os.path.join(out_dir,f'{params['mouse']}_sn_corr_{params['stimulus']}.pkl')}')
        
    
 