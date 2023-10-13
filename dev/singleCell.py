import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import ndimage,stats

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def trial_by_trial(spike_times, event_times, pre, post, bin_size):
    spike_times = np.array(spike_times)
    event_times = np.array(event_times)
    numbins     = np.ceil((pre+post)/bin_size).astype(int)
    bytrial     = np.zeros((len(event_times),numbins))
    
    for j in range(len(event_times)):
        event = event_times[j]
        start = event-pre
        end   = event+post
        
        trial_spikes = spike_times[(spike_times > start) & (spike_times < end)]
        hist         = np.histogram(trial_spikes,bins=numbins)[0]
        bytrial[j,:] = hist
            
    var  = np.var(bytrial,axis=0)/bin_size
    psth = np.mean(bytrial,axis=0)/bin_size
    
    return psth, bytrial, var

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
    
