import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import ndimage,stats
from scipy.signal import savgol_filter

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def cross_from_below(data,threshold,startbin=0):
    window  = data[startbin:]
    crosses = list()
    for i in range(len(window)):
        if window[i-1] < threshold and window[i] >= threshold:
            crosses.append(i+startbin)
            
    return np.array(crosses)

def psth_latency(data,binsize=0.01,pre=0, thresh = 0.5,smooth=False,offset=0):
    if thresh > 1:
        print(f'Cannot use threshold {thresh*100 :.1f}% of max response' )
        return None
    
    if smooth:
        data = savgol_filter(data,5,3)
        
    window_length = len(data)*binsize+binsize
    edges         = np.linspace(-pre,(window_length-pre),len(data))

    startbin  = int(pre/binsize)
    threshold = thresh*data[startbin:].max()
    crossings = cross_from_below(data,threshold,startbin)

    if len(crossings) > 0:
        crossing     = crossings[0] #the first bin above the threshold
        chunk           = np.linspace(data[crossing-1],data[crossing],100)
        bin_crossing = np.array(cross_from_below(chunk,threshold))
        latency      = edges[crossing-1] + (100 - bin_crossing)*(binsize/1000)
        
        if len(latency) > 0:
            return latency[0] - offset
    
    else:
        #print 'response did not exceed threshold: '+str(threshold)+', no latency returned'
        return None

def raster(spike_times, event_times, pre, post):
    spike_times = np.array(spike_times)
    event_times = np.array(event_times)   
    raster      = []
    
    for t in range(len(event_times)):
        event = event_times[t]
        start = event - pre
        end   = event + post
        
        trial_spikes = spike_times[(spike_times >= start) & (spike_times <= end)]
        trial_spikes = trial_spikes-start
        raster.append(trial_spikes)
    
    return raster

def trial_by_trial(spike_times, event_times, pre, post, bin_size):
    buffer      = 0
    # pre         = pre  + buffer
    # post        = post + buffer #Literal voodoo. If this is taken out, all PSTHs will have an empty bin for some reason.
    spike_times = np.array(spike_times).astype(float) + pre
    event_times = np.array(event_times).astype(float)

    numbins  = np.ceil((pre+post)/bin_size).astype(int)-1
    bytrial  = np.zeros((len(event_times),numbins))
    var      = np.zeros((numbins))
    psth     = np.zeros((numbins))
    edges    = np.linspace(-pre,post,numbins)

    for t,time in enumerate(event_times):
        # if len(spike_times[(spike_times >= time-pre)&(spike_times <= time + post)]) > 0:
        if len(np.where(spike_times >= time - pre)[0]) > 0 and len(np.where(spike_times >= time + post)[0]) > 0:
            start = np.where(spike_times >= time - pre)[0][0]
            end   = np.where(spike_times >= time + post)[0][0]
                
            for trial_spike in spike_times[start:end]:
                if float(trial_spike-time)/float(bin_size) <= float(numbins):
                    bytrial[t][int((trial_spike-time)/bin_size)] +=1   
        else:
            continue
        
    bytrial[:,0] = bytrial[:,3]
    bytrial[:,1] = bytrial[:,3]

    var  = np.nanstd(bytrial,axis=0)/bin_size/np.sqrt(len(event_times))
    psth = np.nanmean(bytrial,axis=0)/bin_size

    #constrain your psth to original pre/post size
    bytrial = bytrial[:,(edges >= -(pre-buffer)) & (edges <= post)]
    psth    = psth[     (edges >= -(pre-buffer)) & (edges <= post)]
    var     = var[      (edges >= -(pre-buffer)) & (edges <= post)]
    edges   = edges[    (edges >= -(pre-buffer)) & (edges <= post)] 
        
    return psth, var, edges, bytrial

class SweepMap():
    def __init__(self, spike_times, stim_data, bin_size):
        self.spike_times  = np.array(spike_times)
        self.bin_size     = bin_size
        self.pre          = 0
        
        self.orientations = stim_data.ori.unique()
        self.sweeps       = stim_data.sweep.unique()
        self.raw_output   = {}
        
        lengths = []
        for t in self.sweeps:
            samp = stim_data[stim_data.sweep == t]
            for o in samp.ori.unique():
                start = samp[samp.ori == o].start.values[0]
                end   = samp[samp.ori == o].start.values[-1]
                lengths.append(end-start)
                
        self.post = np.ceil(np.array(lengths).max())
                
        nbins       = np.ceil((self.pre+self.post)/self.bin_size).astype(int)
        psth_all    = np.zeros((len(self.orientations),nbins))
        bytrial_all = np.zeros((len(self.orientations),len(self.sweeps),nbins))
        var_all     = np.zeros((len(self.orientations),nbins))
        
        for i,ori in enumerate(self.orientations):
            ori_sweeps   = stim_data[stim_data.ori == ori]
            
            sweep_starts = []
            sweep_ends   = []
            for t,trial in enumerate(self.sweeps):
                start = ori_sweeps[ori_sweeps.sweep == trial].start.values[0]
                end   = ori_sweeps[ori_sweeps.sweep == trial].start.values[-1]
                sweep_starts.append(start)
                sweep_ends.append(end)
                
            post_ = np.ceil(np.subtract(sweep_ends,sweep_starts)).max()
            nbins = round((self.pre+post_)/self.bin_size)
            
            psth,bt,var             = trial_by_trial(self.spike_times,sweep_starts,self.pre,post_,self.bin_size)
            psth_all[i,:nbins]      = psth
            bytrial_all[i,:,:nbins] = bt
            var_all[i,:nbins]       = var
            
        self.raw_output = {'psth':psth_all,'bytrial':bytrial_all,'var':var_all}
            
    def back_project(self,method=0):
        data = self.raw_output['psth']
        if data.shape[0] != len(self.orientations):
            raise ValueError('Dimension mismatch')
        
        sz   = data.shape[1]

        pad        = int(np.ceil(np.sqrt(2)*sz-sz)/2)
        pad_dat    = np.zeros(int(sz+2*pad))
        pad_dat[:] = np.nan

        cj = np.asmatrix(np.tile(np.arange(-(sz - 1) / 2, (sz + 1) / 2), (sz, 1)))
        ci=cj.H

        map_ = np.ones((sz,sz))
        map_[:,:] = method

        for i in range(len(self.orientations)):
            t = self.orientations[i]*np.pi/180
            pad_dat[pad:pad+sz] = data[i]
            rcj = np.asmatrix(np.round(cj*np.cos(t)-ci*np.sin(t)+np.ceil(pad+sz/2))-1,dtype='int16')
            dat_array = pad_dat[rcj] 

            if method:
                map_ *= dat_array
            else:
                map_ += dat_array
                
        if method == 1:
            map_ /= len(self.orientations)
        elif method == 2:
            s           = np.sign(map_)
            isnan       = np.isnan(map_)
            map_[isnan] = 1
            map_        = np.abs(map_) ** (1 / len(self.orientations)) * s
            map_[isnan] = np.nan

        return map_
    
     
class rCorr():
    def __init__(self,spike_times,stim_times,signal,taus=np.linspace(-0.01,0.28,30),exclusion=None):
        self.signal      = np.array(signal)
        self.stim_times  = np.array(stim_times)
        self.spike_times = np.array(spike_times)
        self.taus        = np.array(taus)
        
        self.start       = stim_times[0]
        self.end         = stim_times[-1]-0.06
        
        self.stim_spikes = self.spike_times[(self.spike_times >= self.start) & (self.spike_times < self.end)]
        self.spikes_adj  = self.stim_spikes[:,np.newaxis] - self.taus
        
        self.spikes_adj  = self.spikes_adj.T
        
        if exclusion is not None: 
            # Check if there are time periods we should ignore (eye closing, stim issues, etc.)
            for i in exclusion:
                ex1 = find_nearest(self.stim_spikes,i[0])
                ex2 = find_nearest(self.stim_spikes,i[1])
                self.stim_spikes = np.delete(self.stim_spikes,np.arange(ex1,ex2))
                
                
    def sta(self):
        output = np.zeros(((len(self.taus),) + self.signal[0].shape))
        output[:] = np.nan
        
        for i,tau in enumerate(self.taus):
            avg = np.zeros(self.signal[0].shape)
            count = 0
            for spike in self.stim_spikes:
                index = (np.where(self.stim_times > (spike - tau))[0][0]-1)
                avg += self.signal[index]
                count+=1
                    
            output[i,:,:] = avg/count
            
        return output,self.taus
            
        
    def sta2(self):
        output = np.zeros((len(self.taus),*self.signal.shape[1:]))

        for i, spikes in enumerate(self.spikes_adj):
            triggered = []
            for ts in range(len(self.stim_times)):
                nspikes =  len(spikes[(spikes >= self.stim_times[ts-1]) & (spikes < self.stim_times[ts])])
                triggered.append(nspikes)
            output[i] = np.average(self.signal,axis=0,weights=triggered)
            
        return output,self.taus   
    
