import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from itertools import combinations
import platform

from numpy.fft import fft,fftshift,ifft,ifftshift
from scipy.stats.mstats import gmean

def nextpow2(n):
    """get the next power of 2 that's greater than n"""
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i

def triangle_fx(n_t):
    t      = np.arange(-(n_t-1),(n_t-1))
    theta  = n_t-np.abs(t)

    NFFT   = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])
    
    return NFFT, target,theta

def xcorrfft(a,b,target,NFFT):
    """Cross Correlation using FFT
        Conjugation and multiplication is a mathematical trick to compute sin^2+cos^2 (i.e. hypotenuse (magnitude) squared). The result will be the spectrum of the autocorrelation function, which you can then ifft to get the autocorrelation function itself.
    Args:
        a    (array) : signal 1
        b    (array) : signal 2
        NFFT (int)   : length of output

    Returns:
        CCG: _description_
    """
    CCG_ = np.abs(fftshift(ifft(np.multiply(fft(a,NFFT), np.conj(fft(b,NFFT))))))
    if len(CCG_.shape)==3:
        CCG = np.squeeze(np.nanmean(CCG_[:,:,target],axis=1))
    else:
        temp = CCG_[target]
        CCG  = temp[int(len(target)/2)-100: int(len(target)/2)+100]
    
    return CCG

#From https://github.com/jiaxx/ccg_jitter/

def jitter(data, window):
    """
    Jittering multidemntational logical data where 
    0 means no spikes in that time bin and 1 indicates a spike in that time bin.
    
    data: [time*trial*condition]
    l   : jitter window [ms] (assuming 1ms binsize)
    """
    psth     = data.mean(axis=1) #average for each condition
    length   = len(data)         #number of time bins
    n_trials = data.shape[1]
    n_cond   = data.shape[2]
    
    # if number of bins cannot be divided by the jitter window
    if length%window:
        data[length:length+(-length%window),:,:] = 0
        psth[length:length+(-length%window),:]   = 0
        
    if n_cond > 1:
        dataj = np.squeeze(np.reshape(data,[window,length//window,n_trials,n_cond],order='F').sum(axis=0))
        psthj = np.squeeze(np.reshape(psth,[window,length//window,n_cond],order='F').sum(axis=0))
        
    else:
        dataj = np.squeeze(np.reshape(data,[window,length//window,n_trials],order='F').sum())
        psthj = np.reshape(data,[window,length//window],order='F').sum()
        
    if length == window:
        dataj = dataj.reshape((1,window,dataj.shape[1],),order='F')
        psthj = psthj.reshape((1,window,),order='F')
        
    psthj           = psthj.reshape((psthj.shape[0],1,psthj.shape[1],),order='F')
    psthj[psthj==0] = 10e-10

    corr = dataj/np.tile(psthj,[1, dataj.shape[1], 1])
    corr = corr.reshape((1,*corr.shape[:3]),order='F')
    corr = np.tile(corr,[window, 1, 1, 1])
    corr = corr.reshape((np.multiply(*corr.shape[:2]),corr.shape[2],corr.shape[3]),order='F')
    
    psth = psth.reshape((len(psth),1,psth.shape[1]),order='F')
    output = np.tile(psth,[1, corr.shape[1], 1])*corr

    output = output[:length,:,:]
    return output

class xCorr():
    def __init__(self,iu1,iu2,trial_info,units_df=None,frs=(),**kwargs):
        """_summary_

        Args:
            iu1 (int): first unit index (index must match trial_info and units_df if frs not provided)
            iu2 (int): second unit index (index must match trial_info)
            trial_info (ndarray): Ideally, shaped (units,conditions,trials,time). Can also accommodate (units,trials*conditions,time)
            units_df (data frame, optional): units data frame containing ks info. Defaults to None.
            frs (tuple, optional): Without the entire units_df, you can also extract the fr column or just extract the two relevant frs. Defaults to ().
        """
        self.temp1 = trial_info[iu1]
        self.temp2 = trial_info[iu2]
        
        if units_df:
            self.ksfr1 = units_df.iloc[iu1].fr
            self.ksfr2 = units_df.iloc[iu2].fr
        else:
            if len(frs) > 2:
                self.ksfr1 = frs[iu1]
                self.ksfr2 = frs[iu2]
            if len(frs) == 2:
                self.ksfr1 = frs[0]
                self.ksfr2 = frs[1]
        if self.ksfr1 < 2 or self.ksfr2 < 2:
            print('Firing rate too low')
            return
                
        self.gfr = np.sqrt(self.ksfr1*self.ksfr2)        
        self.n_t = self.temp1.shape[-1]
        
        if len(self.temp1.shape) < 3 and 'n_cond' in kwargs:
            self.ncond = kwargs['n_cond']
            self.temp1 = self.temp1.reshape(self.ncond,self.temp1.shape[0]//self.ncond,self.n_t)
            self.temp2 = self.temp1.reshape(self.ncond,self.temp1.shape[0]//self.ncond,self.n_t)
            
        self.n_trials = self.temp1.shape[1]
        self.NFFT, self.target, self.theta = triangle_fx(self.n_t)
        
        self.fr1 = np.squeeze(np.mean(np.sum(self.temp1,axis=2), axis=1))
        self.fr2 = np.squeeze(np.mean(np.sum(self.temp2,axis=2), axis=1))
        
    def ccg(self):
        tempccg = xcorrfft(self.temp1,self.temp2,self.target,self.NFFT)
        ccg_    = np.squeeze(np.nanmean(tempccg[:,:,self.target],axis=1))
        ccg_    = ccg_[int(len(self.target)/2)-100: int(len(self.target)/2)+100]
        return ccg_
    
    def ccg_gmean(self):
        tempccg = self.ccg()
        ccg_    = tempccg/np.sqrt(self.fr1*self.fr2)
        return ccg_
    
    def ccg_jitter(self,window=25):
        temp1_   = np.rollaxis(np.rollaxis(self.temp1,2,0),2,1) #time*trial*orientation
        temp2_   = np.rollaxis(np.rollaxis(self.temp2,2,0),2,1)
        ttemp1  = jitter(temp1_,window)
        ttemp2  = jitter(temp2_,window)
        
        tempjitter = xcorrfft(np.rollaxis(np.rollaxis(ttemp1,2,0), 2,1),np.rollaxis(np.rollaxis(ttemp2,2,0), 2,1),self.target,self.NFFT)
        
        gmean_ = np.multiply(np.tile(np.sqrt(self.ksfr1*self.ksfr2), (len(self.target), 1)), np.tile(self.theta.T.reshape(len(self.theta),1),(1,self.n_cond)))       
        
            
                
            
