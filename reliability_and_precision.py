import numpy as np
import random
from scipy import stats
from scipy.signal import boxcar,convolve,correlate,resample,argrelextrema
from scipy.cluster.vq import kmeans,kmeans2
from scipy.stats import pearsonr
from dlab import cleanAxes
from dlab import psth_and_raster as psth_

def smooth_boxcar(data,boxcar_size):
    smoothed = convolve(data,boxcar(int(boxcar_size)))/boxcar_size
    smoothed = smoothed[int(boxcar_size/2):len(data)+int(boxcar_size/2)]
    return smoothed

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_peak_waveform_from_template(template):
    max = 0
    ind=0
    peak = np.zeros(np.shape(template.T)[0])
    for i,wv in enumerate(template.T):
        if np.max(np.abs(wv)) > max:
            max = np.max(np.abs(wv))
            ind = i
            peak = wv
    return peak

def precision(spike_times,stimulus_times,boxcar_size = 5,precision_type='first',pre=0.,post=15.,binsize=0.01,threshold=0.05,find_events=True):
# precision - jitter of first spike, jitter of total spikes
    if find_events:
        smoothed,all_events,events = get_events(spike_times,stimulus_times,boxcar_size = boxcar_size,threshold=threshold,pre=pre,post=post,binsize=binsize)
    else:
        events = [0.]
    #print events
    #get spikes for each event
    event_precision = []
    for i,event in enumerate(events):
        all_spikes = []
        first_spikes = []
        for trial in stimulus_times:
            event_start = trial + event
            if i == len(events):
                if event[i+1] - event > 0.05:
                    event_end = trial + event[i+1]
                else:
                    event_end = trial + event + 0.05
            else:
                event_end = trial + event + 0.05
            indices = np.where((spike_times > event_start) & (spike_times < event_end))[0] 

            first_spike = spike_times[np.where(spike_times>event_start)[0][0]]-event_start
            #find jitter of first spikes
            if first_spike < .1: #arbitrary cutoff because we don't trust identified events that don't have a spike within 100 msec
                first_spikes.extend([first_spike])

            #find jitter of all spikes
            event_spikes = np.array(spike_times[indices])-event_start
            all_spikes.extend(event_spikes)
  
        if precision_type =='first':
            #event_precision.extend([np.median(np.array(first_spikes))])
            #event_precision.extend([np.median(np.array(first_spikes)-np.min(first_spikes))])
            all_event_spikes = np.sort(np.array(first_spikes))
        else:
            all_event_spikes = np.sort(np.array(all_spikes).flatten())
            
        #print all_event_spikes
        first_half = all_event_spikes[:np.where(all_event_spikes<np.median(all_event_spikes))[0][-1]]
        second_half = all_event_spikes[np.where(all_event_spikes>np.median(all_event_spikes))[0][0]:]
        event_precision.extend([np.median(second_half)-np.median(first_half)])
    
    event_precision = np.array(event_precision)
    event_precision = event_precision[~np.isnan(event_precision)] * 1000.
    
    return (np.mean(event_precision),np.std(event_precision),event_precision)

def get_binarized(spike_times,stimulus_times,pre=0.,post=15.,convolve=0.):
    bytrial = psth_.raster(spike_times,stimulus_times,pre=pre,post=post,timeDomain=True,output='data')
    binarized = []
    for trial in bytrial:
        binarized_ = np.zeros(int((post-pre)*1000))#use 1-msec bins
        for spike in trial:
            if spike > pre and spike < post:
                binarized_[int(np.floor(spike*1000))] = 1
        if convolve > 0.001 :
            binarized_=smooth_boxcar(binarized_,convolve)
        binarized.append(binarized_)   
    return binarized

def get_binned(spike_times,stimulus_times,binsize,pre=0.,post=15.,convolve=0.):
    bytrial = psth_.raster(spike_times,stimulus_times,pre=pre,post=post,timeDomain=True,output='data')
    # print(np.array(bytrial).shape())
    binarized = []
    for trial in bytrial:
        binarized_ = np.zeros(int((post-pre)*1000/(1000*binsize)+1))#use binsize msec bins
        for spike in trial:
            if spike > pre and spike < post:
                binarized_[int(np.floor((spike-pre)*1000/(1000*binsize)))] += 1
        if convolve > 0.001 :
            binarized_=smooth_boxcar(binarized_,convolve)
        binarized.append(binarized_)   
    return binarized

def reliability(spike_times,stimulus_times,binsize,pre=0.,post=15.):
#reliability - how reproducible a spike train is over trials, at the msec level
    binarized=get_binned(spike_times,stimulus_times,binsize,pre=pre,post=post)
    sum = 0
    for c,i in enumerate(binarized):
        for c2 in np.arange(c+1,np.shape(binarized)[0],1):
            j = binarized[c2][:]
            if np.sum(i) > 0 and np.sum(j) > 0:
                sum += np.inner(i,j) / np.inner(np.linalg.norm(i),np.linalg.norm(j))
            else:
                sum+=0

    #return (2 / float(np.shape(binarized)[0]) * (np.shape(binarized)[0] - 1)) * sum,binarized
    return sum / (np.shape(binarized)[0]/2.)#,binarized

def fano(spike_times,stimulus_times,pre=0.,post=15.,binsize=0.01,boxcar_size = 5,counting_window=0.3,threshold=0.2,by_event=False):
    if by_event:
        smoothed,all_events,events = get_events(spike_times,stimulus_times,boxcar_size = boxcar_size,threshold=threshold,pre=pre,post=post,binsize=binsize)
    else:
        events = [0]
    fanos=[]
    for i,event in enumerate(events):
        counts=[]
        for trial in stimulus_times:
            event_start = trial + event
            event_end = trial + event + counting_window
            indices = np.where((spike_times > event_start) & (spike_times < event_end))[0] 
            
            counts.append(len(indices))

        fanos.append(np.std(counts)**2 / np.mean(counts))
    return np.median(fanos),counts, fanos

def get_events(spike_times,stimulus_times,threshold=.05,boxcar_size = 15,pre=0.,post=15.,binsize=0.001):
    (edges,psth,variance) =  psth_.psth_line(spike_times,
                                             stimulus_times,
                                             pre=pre,post=post,binsize=binsize,output='p',timeDomain=True)
    numbins = int((post-pre) /  binsize)

    # first, find events:
    smoothed = smooth_boxcar(psth[:numbins],int(boxcar_size))
    minima = np.where(np.r_[True, smoothed[2:] > smoothed[:-2]] & np.r_[smoothed[:-2] > smoothed[2:], True]==True)[0]#from http://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
   
    #print np.max(psth)
    threshold = threshold * np.max(psth)
    good_minima = []
    #print threshold
    for i,minimum in enumerate(minima[:-1]):
        if minima[i+1]*binsize - minima[i]*binsize > 0.3:
            num_bins_after_minimum = 0.3 / binsize
        else:
            num_bins_after_minimum = minima[i+1] - minima[i]
        try:
            if np.max(psth[minimum:minimum+num_bins_after_minimum]) > threshold:
                good_minima.extend([minimum])
        except:
            pass
            # print np.shape(psth)
            # print minimum
            # print num_bins_after_minimum
            # print threshold
            # return np.nan,np.nan,np.nan
            # 

    return smoothed,minima*binsize,np.array(good_minima)*binsize - (boxcar_size/2.)*binsize

def entropy(spike_times,stimulus_times,wordlength,binsize=0.001,pre=0.,post=15.):
    binarized=get_binarized(spike_times,stimulus_times,pre=pre,post=post)

    # create words of length wordlength 
    entropies_per_time=[]
    for t in range(len(binarized[0])):
        words = []
        for trial in binarized:
            if t<len(trial) - wordlength: #cutoff of the end of each trial because there aren't enough bins left to make the word
                word = trial[t:t+wordlength]
                words.append(word)

        #make a distribution of the words
        p = {}
        #find all the words that actually occured. and the frequency of their occurence
        #from http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
        # and http://stackoverflow.com/questions/33235135/count-occurrences-of-arrays-in-multidimensional-arrays-in-python
        words = np.array(words)
        b = np.ascontiguousarray(words).view(np.dtype((np.void, words.dtype.itemsize * words.shape[1])))
        _, idx,counts = np.unique(b, return_index=True,return_counts=True)
        possible_words = words[idx]
        p = dict(zip([str(word) for word in possible_words],counts/float(np.shape(np.array(words))[0])))
        print('all the words that occured:'+str(np.shape(np.array(words))[0]))

        # calculate entropy as in reinagel and reid 2000; kumbhani et al., 2007
        sum = 0
        for word in possible_words:
            sum +=  p[str(word)] * np.log2(p[str(word)])
        H = (-1 / (wordlength*binsize)) * sum
        entropies_per_time.append(H)
        
    return np.mean(entropies_per_time)

def fano2(a):
# computes the Fano factor , variance**2/mean for an input matrix a with dimensions n x m, 
# where n is the number of trials and m is the number of bins in the response
    return np.std(a,axis=0)**2/np.mean(a,axis=0)

def ccmax(a):
# computes the cc_max for an input matrix a with dimensions n x m, 
# where n is the number of trials and m is the number of bins in the response
    ntrials = a.shape[0]
    corr_matrix = np.empty((ntrials,ntrials))
    for i in range(ntrials):
        for j in range(ntrials):
            r,p = pearsonr(a[i,:],a[j,:])
            corr_matrix[i,j] = r

    inds = np.triu_indices(ntrials, k=1)
    upper = corr_matrix[inds[0],inds[1]]
    return np.nanmean(upper)

def mutual_information(spike_times,stimulus_times,wordlength,binsize=0.001,pre=0.,post=15.,method='poisson'):
    indices = np.where((spike_times > stimulus_times[0]) & (spike_times < stimulus_times[-1]))[0] 
    if method == 'poisson':
        reference_spikes = possion_times(np.array(spike_times)[indices])
    if method == 'shift':
        reference_spikes = shifted_times(np.array(spike_times)[indices],stimulus_times)
    total = entropy(reference_spikes,stimulus_times,wordlength,binsize=0.001,pre=0.,post=15.)
    noise = entropy(np.array(spike_times),stimulus_times,wordlength,binsize=0.001,pre=0.,post=15.)
    return total,noise, total-noise, (total-noise) / (post - pre)

def z_value(spike_times,stimulus_times,binsize,pre=0.,post=15.,method='shift'):
#based on reinagel and reid, 2000
# their definition, sightly modified for easier notation here
# Z(binsize) = limI(L,binsize) - I(L=1,binsize)
# "The term I(L=1,binsize) represents the estimate of information rate that would be obtained on the approximation that time bins are statistically independent within the spike train."
    lim_I = mutual_information(spike_times,stimulus_times,wordlength,binsize=binsize,pre=0.,post=15.,method='poisson')
    I1 = mutual_information(spike_times,stimulus_times,1,binsize=binsize,pre=0.,post=15.,method='poisson')
    return z

def possion_times(spike_times):
#given an input spike train, make the times a rate-matched Poisson process
    rate = len(np.array(spike_times)) / (np.array(spike_times)[-1] - np.array(spike_times)[0])
    t = np.array(spike_times)[0]
    poiss = [t]
    for i in range(len(spike_times)-1):
        t+=random.expovariate(rate)
        poiss.append(t)
    return np.array(poiss)

def shuffled_times(spike_times,window):
#given an input spike train, shuffle the spike times preserving the structure within the window
    return spike_times

def shifted_times(spike_times,stimulus_times):
#given an input spike train and stimulus times, 
#shift the spike times in each interval by an random amount, with wrapping
    shifted_times = np.zeros(len(spike_times))
    for i,start in enumerate(stimulus_times[:-1]):
        indices = np.where((spike_times > start) & (spike_times <= stimulus_times[i+1]))[0] - 1
        times = spike_times[indices]
        offset = np.floor(np.random.rand() * (stimulus_times[i+1] - start))
        offset_times = times + offset + start - i*(stimulus_times[i+1] - start)
        wrapped_times = np.array([b if b < stimulus_times[i+1] else b-(stimulus_times[i+1]-start) for b in offset_times])
        #print str(start)+' '+str(offset)+' '+str(offset_times)+' '+str(wrapped_times)

        shifted_times[indices] = wrapped_times
    indices = np.where(spike_times > stimulus_times[-1])[0]
    times = spike_times[indices]
    offset = np.random.rand() * (spike_times[-1] - stimulus_times[-1])
    offset_times = times + offset
    wrapped_times = np.array([b if b < spike_times[-1] else b-(spike_times[-1]-stimulus_times[-1]) for b in offset_times])
    shifted_times[indices] = wrapped_times
    return np.sort(shifted_times) - stimulus_times[0]


# def Rjitter_pair(spike_times, other_spike times):
#     spearmanr(spike_times,other_spike,nan_policy='omit')
    