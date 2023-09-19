import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx
    
class TrialByTrial:
    def __init__(self, spike_times, event_times, pre, post, bin_size):
        
        numbins = int((pre+post)/bin_size)
        
        self.bytrial = np.zeros((len(event_times),numbins))
        for j in range(len(event_times)):
            event = event_times[j]
            start = event-pre
            end   = event+post
            
            trial_spikes = spike_times[(spike_times > start) & (spike_times < end)]
            hist = np.histogram(trial_spikes,bins=numbins)[0]
            self.bytrial[j,:] = hist
                
        self.var = np.nanvar(self.bytrial,axis=0)/bin_size
            
        self.psth = np.nanmean(self.bytrial,axis=0)/bin_size
        
        
class PSTH(TrialByTrial):
    def __init__(self, spike_times, event_times, pre, post, bin_size):
        super().__init__(spike_times, event_times, pre, post, bin_size)
        
        
class Raster(TrialByTrial):
    def __init__(self, spike_times, event_times, pre, post):
        self.bin_size = 1/30000 #30kHz sampling rate
        super().__init__(spike_times, event_times, pre, post,bin_size = self.bin_size)
        
        self.spike_times = spike_times
    
        
# class Tuning(TrialbyTrial):
#     def __init__(self, spike_times, event_times, pre, post, bin_size):
#         super().__init__(spike_times, event_times, pre, post, bin_size)
        
        
        
class rCorr():
    def __init__(self,spike_times,stim_times,signal,taus=np.linspace(-0.01,0.28,0.30),exclusion=None):
        self.stim_data   = np.asarray(signal)
        self.stim_times  = np.asarray(stim_times)
        self.taus        = np.asarray(taus)
        
        self.start       = stim_times[0]
        self.end         = stim_times[-1]-0.06
        
        self.stim_spikes = spike_times[(spike_times > self.start) & (spike_times < self.end)]
        
        if exclusion is not None: 
            # Check if there are time periods we should ignore (eye closing, stim issues, etc.)
            for i in exclusion:
                ex1 = find_nearest(self.stim_spikes,i[0])
                ex2 = find_nearest(self.stim_spikes,i[1])
                self.stim_spikes = np.delete(self.stim_spikes,np.arange(ex1,ex2))
            
        
    def sta(self):
        #TODO: Figure out if it's necessary to find frame  before closest frame preceding spike - tau
        # That adds 0.05s to the STA, which is a lot
        
        #TODO: Edit code to accommodate 1D signals
        
        output = np.full((len(self.taus),*self.stim_data.shape[1:]),np.nan)
        
        # turn spikes-taus into 2d array
        spikes_adj = self.stim_spikes - self.taus[:,np.newaxis]
        
        for i in range(len(spikes_adj)):
            avg         = np.full(self.signal.shape[1:],np.nan)
            spike_count = 0
            for spike in spikes_adj[i]:
                idx = np.where(spike >= self.stim_times)[0].argmin() #closest frame preceding spike - tau
                avg += self.stim_data[idx]
                spike_count += 1
                
            avg /= spike_count
            output[i] = avg
            
        return output,self.taus
    
    def stc(self):
        pass
    
# class fCorr():
#     def __init__(spike_times, signal, bin_size):