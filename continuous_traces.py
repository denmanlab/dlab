import numpy as np
import scipy, os
from scipy.signal import butter,lfilter
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.pyplot import mlab
import xml.etree.ElementTree
samplingRate=30000.
#=================================================================================================
#------------operations on continuous traces-------------------------------------
#=================================================================================================

npix_p3_reference_channels = np.array([ 36,  75, 112, 151, 188, 227, 264, 303, 340, 379])
npix_p2_reference_channels = np.array([1,18,33,50,65,82,97,114,99])
skip_channels = npix_p3_reference_channels #default to phase 3 reference channels
def get_chunk(mm,start,end,channels,sampling_rate=30000,remove_offset=False):
	chunk = mm[int(start*sampling_rate*int(channels)):int(np.floor(end*sampling_rate*(int(channels))))]
	chunk = np.reshape(chunk,(int(channels),-1),order='F')  * 0.195
	# print(np.shape(chunk))
	if remove_offset:
		for i in np.arange(channels):
			chunk[i,:] = chunk[i,:] - np.mean(chunk[i,:])
	return chunk

#filter a bit of continuous data. uses butterworth filter.
def filterTrace(trace, low, high, sampleHz, order):
    low = float(low)
    high = float(high)
    nyq = 0.5 * sampleHz
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = lfilter(b, a, trace)
    return filtered
    
#developmental filter version. not used.
def filterTrace_hard(trace, low, high, sampleHz, order):
    low = float(low)
    high = float(high)
    nyq = 0.5 * sampleHz
    low = low / nyq
    high = high / nyq
    scipy.signal.band_stop_obj()
    b, a = butter(order, [low, high], btype='band')
    filtered = lfilter(b, a, trace)
    scipy.signal.lfilter()
    return filtered

#wrapper for filtering continous data of different forms.
#data can be a single continuous trace, a dictionary containing a key called 'data' whose value is a continous trace, or a dictionary of traces, or a dicit
def filtr(data,low, high, sampleHz, order):
    if type(data) is dict:
        if 'data' in data.keys():
            return filterTrace(data['data'],low, high, sampleHz, order)
        else:
            out = {}
            for i,key in enumerate(data.keys()):
                out[key] = data[key]
                out[key]['data']= filterTrace(data[key]['data'],low, high, sampleHz, order)
            return out
    else:
        return filterTrace(data,low, high, sampleHz, order)

#notch filter a continous trace by filtering in a narrow range and subtracting that from the input trace.    
def notch(data,freq, sampleHz):
    order = 1
    low = freq-2
    high = freq +2
    if type(data) is dict:
        if 'data' in data.keys():
            return data['data'] - filterTrace(data['data'],low, high, sampleHz, order)
        else:
            out = {}
            for i,key in enumerate(data.keys()):
                out[key] = data[key]
                out[key]['data']= data[key]['data'] - filterTrace(data[key]['data'],low, high, sampleHz, order)
            return out
    else:
        return data - filterTrace(data,low, high, sampleHz, order)

#average a continuous trace around a set of timestamps 
def average_trials(data,timestamps,window,sampleFreq=25000.):
    alltrials = np.zeros((len(timestamps),window*sampleFreq))
    average = np.zeros(window*sampleFreq)
    skipped = 0
    for i,onset in enumerate(timestamps):
        average += data[onset:onset+window*sampleFreq]#-np.mean(data[onset:onset-500])
        alltrials[i,:] = data[onset:onset+window*sampleFreq]#-np.mean(data[onset:onset-500])
        
#        if np.max(np.abs(data[onset:onset+window*sampleFreq]-np.mean(data[onset:onset+5000]))) < 40000000.0:
#            average += data[onset:onset+window*sampleFreq]-np.mean(data[onset:onset+5000])
#            alltrials[i,:] = data[onset:onset+window*sampleFreq]-np.mean(data[onset:onset+5000])
#        else:
#            skipped += 1
#            print 'skipped trial: '+str(i+1)
#            alltrials[i,:] = data[onset:onset+window*sampleFreq]-np.mean(data[onset:onset+5000])
    return alltrials,average/float(len(timestamps-skipped))   

#average all continuous traces in an array around a set of timestamps 
def average_trials_array(data,timestamps,window,output='avg'):
    avgs = {}
    alltrials={}
    for i,key in enumerate(data.keys()):
    	if 'data' in data[key].keys():
	        avgs[key]={}
	        alltrials[key]={}
	        alltrials[key]['data'],avgs[key]['data'] = average_trials(data[key]['data'],timestamps,window)
    if output == 'trials':
        return alltrials
    if output == 'both':
        return (alltrials,avgs)
    if output=='avg':
        return avgs
	

#note: this CSD code does not work! -dan
def CSD_1D(data,channelmap=[],prefix='100_CH',point=1000):
    if channelmap == []:
        channelmap = data.keys()
    elec_pos = []
    pots=[]    
    for i,key in enumerate(channelmap[0]):
        key = prefix+str(key).replace(prefix,'')
        pots.append([data[key]['data'][point]])
        elec_pos.append([(i+i)/2])
    pots=np.array(pots)
    elec_pos=np.array(elec_pos)
    params = {
        'xmin': 0,
        'xmax': 65.0,
        'source_type': 'step',
        'n_sources': 64,
        'sigma': 0.1
        }
    k = KCSD(elec_pos, pots, params)
    k.estimate_pots()
    k.estimate_csd()
    k.plot_all()

#note: this CSD code does not work! -dan    
def CSD_1D_time(data,channelmap=[],prefix='100_CH',point=1000):
    if channelmap == []:
        channelmap = data.keys()
    numPoints = len(data[data.keys()[0]]['data'])
    out_csd = np.zeros((len(data.keys()),numPoints))
    out_pots = np.zeros((len(data.keys()),numPoints))
    for point in range(numPoints):
        print(point)
        elec_pos = []
        pots=[]    
        for i,key in enumerate(channelmap[0]):
            key = prefix+str(key).replace(prefix,'')
            pots.append([data[key]['data'][point]])
            elec_pos.append([i+i])
        pots=np.array(pots)
        elec_pos=np.array(elec_pos)
        params = {
            'xmin': 0,
            'xmax': 130.0,
            'source_type': 'step',
            'n_sources': 128,
            'sigma': 0.2,
            }
        k = KCSD(elec_pos, pots, params)
        k.estimate_pots()
        k.estimate_csd()
        out_csd[0:np.shape(k.solver.estimated_csd)[0],point]= k.solver.estimated_csd[:,0]
        out_pots[0:np.shape(k.solver.estimated_pots)[0],point]= k.solver.estimated_pots[:,0]
    return out_csd,out_pots
        #k.plot_all()

def etree_to_dict(t):
    d = {t.tag : list(map(etree_to_dict, t.getchildren()))}
    d.update(('@' + k, v) for k, v in t.attrib.items())
    d['text'] = t.text
    return d

def get_channel_count(path,from_channel_map = True,from_templates=False):
	d = etree_to_dict(xml.etree.ElementTree.parse(os.path.join(path,'settings.xml')).getroot())
	chs =0
	if from_templates:
		return np.load(open(os.path.join(path,'templates.npy'))).shape[-1]
	if d['SETTINGS'][1]['SIGNALCHAIN'][0]['@name'] == 'Sources/Neuropix':
		for info in d['SETTINGS'][1]['SIGNALCHAIN'][0]['PROCESSOR'][:385]:
			if 'CHANNEL' in info.keys():
				if info['CHANNEL'][0]['@record'] == '1':
					chs +=1
		return chs
	if d['SETTINGS'][1]['SIGNALCHAIN'][0]['@name'] == 'Sources/Rhythm FPGA':
		if from_channel_map:
			for nm in d['SETTINGS'][1]['SIGNALCHAIN']:
				name = nm['@name']
				if name == 'Filters/Channel Map':
					#chs = np.shape(d['SETTINGS'][1]['SIGNALCHAIN'][0]['PROCESSOR'][0]['CHANNEL_INFO'])[0]
					for info in nm['PROCESSOR']:
						if 'CHANNEL' in info.keys():
							if info['CHANNEL'][0]['@record'] == '1':
								chs +=1
		else:
			for info in d['SETTINGS'][1]['SIGNALCHAIN'][0]['PROCESSOR'][:385]:
				if 'CHANNEL' in info.keys():
					if info['CHANNEL'][0]['@record'] == '1':
						chs +=1
		return chs

#returns the root mean squared of the input data    
def RMS(data,start=0,window=0,despike=False):
	start = start * samplingRate# sampling rate
	if window == 0:
		window = len(data)
	else:
		window = window * samplingRate # sampling rate
	#chunk = filterTrace(data[start:start+window], 70, 6000, 25000, 3)[200:window]
	chunk = data[int(start):int(start)+int(window)] - np.mean(data[int(start):int(start)+int(window)])
	if despike:
		chunk = despike_trace(chunk,threshold=180)
	return np.sqrt(sum(chunk**2)/float(len(chunk)))

def despike_trace(trace,threshold_sd = 2.5,**kwargs):
	if 'threshold' in kwargs.keys():
		threshold = kwargs['threshold']
	else:
		threshold = np.mean(trace)+threshold_sd*np.std(trace)
		
	spike_times_a = mlab.cross_from_below(trace,threshold)
	spike_times_b = mlab.cross_from_below(trace,-1*threshold)
	for spike_time in np.concatenate((spike_times_b,spike_times_a)):
		if spike_time > 30 and spike_time < len(trace)-30:
			trace[spike_time - 20:spike_time + 20] = 0#np.random.uniform(-1*threshold,threshold,60)
	return trace

def spikeamplitudes_trace(trace,threshold_sd = 3.0,percentile = 0.9,**kwargs):
	if 'threshold' in kwargs.keys():
		threshold = kwargs['threshold']
	else:
		threshold = np.mean(trace)+threshold_sd*np.std(trace)
		
	spike_times_a = mlab.cross_from_below(trace,threshold)
	amps=[]
	for spike_time in spike_times_a:
		if spike_time > 30 and spike_time < len(trace)-30:
			amps.extend([np.max(np.abs(trace[spike_time-30:spike_time+30]))])
	if not len(amps) > 10:
		amps= [0]
	return np.sort(amps)[int(len(amps)*percentile)]# / 5.0

#returns the peak to peak range of the input data     
def p2p(data,start=0,window=0):
    start = start * samplingRate# sampling rate
    if window == 0:
        window = len(data)
    else:
        window = window * samplingRate # sampling rate
    chunk = data[start:start+window]
    return np.max(chunk)-np.min(chunk)
    
#computes a power spectrum of the input data
#optionally, plot the computed spectrum
def b(data,start=0,window=0,plot=False,ymin=1e-24,ymax=1e8,title='',samplingRate=2500):
    start = start * samplingRate# sampling rate
    if window == 0:
        window = len(data)
    else:
        window = window * samplingRate # sampling rate
    chunk = data[start:start+window]/1e6
    ps = np.abs(np.fft.fft(chunk))**2
    time_step = 1. / samplingRate
    freqs = np.fft.fftfreq(chunk.size, time_step)
    idx = np.argsort(freqs)
    ps = scipy.signal.savgol_filter(ps,5,3)
    if plot:
        plt.plot(freqs[idx], ps[idx]);
        plt.xlim(xmin=0.01);
        plt.ylim(ymin=ymin,ymax=ymax)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$power\/density\/\frac{V^2}{Hz}$',color='k',fontsize=18)
        plt.xlabel(r'$frequency,\/ Hz$',color='k',fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)#;plt.locator_params(axis='y',nbins=6)
        plt.title(title)
    return (freqs[idx], ps[idx])

def periodogram(data,start=0,window=0,plot=False,ymin=1e-24,ymax=1e8,title='',samplingRate=2500):
    start = start * samplingRate# sampling rate
    if window == 0:
        window = len(data)
    else:
        window = window * samplingRate # sampling rate
    chunk = data[start:start+window]
    f,pXX = scipy.signal.periodogram(chunk,samplingRate,nfft=samplingRate)
    pXX = scipy.signal.savgol_filter(pXX,3,1)
    if plot:
        plt.plot(f, pXX);
        plt.xlim(xmin=0.5);
        plt.ylim(ymin=ymin,ymax=ymax)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$power\/density\/\frac{V^2}{Hz}$',color='k',fontsize=18)
        plt.xlabel(r'$frequency,\/ Hz$',color='k',fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)#;plt.locator_params(axis='y',nbins=6)
        plt.title(title)
    return (f, pXX)

def welch_power(data,samplingRate=2500,start=0,window=0,plot=False,ymin=1e-24,ymax=1e8,title=''):
    start = start * samplingRate# sampling rate
    if window == 0:
        window = len(data);print(window)
    else:
        window = window * samplingRate # sampling rate
    chunk = data[start:start+window]
    f,pXX = scipy.signal.welch(chunk,samplingRate,nfft=samplingRate/2)
    #pXX = scipy.signal.savgol_filter(pXX,3,1)
    if plot:
        plt.plot(f, pXX);
        plt.xlim(xmin=0.01);
        plt.ylim(ymin=ymin,ymax=ymax)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$power\/density\/\frac{V^2}{Hz}$',color='k',fontsize=18)
        plt.xlabel(r'$frequency,\/ Hz$',color='k',fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)#;plt.locator_params(axis='y',nbins=6)
        plt.title(title)
    return (f, pXX)

#measure the cross-spectral coherence between two traces. 
def coherence(x,y,samplingRate = 30000,returnval=None):
    spectrum, frequencies = mlab.cohere(x,y,Fs=float(samplingRate),NFFT=int(samplingRate)/5)
    if returnval:
        if type(returnval) is float:
            return np.interp(returnval,frequencies,spectrum)
        if type(returnval) is tuple:
            return np.trapz(spectrum[np.where(frequencies==returnval[0])[0]:np.where(frequencies==returnval[1])[0]],dx=5.0)  
    else:
        return (spectrum, frequencies)

def get_surface_channel_spikeband(path,start=2.,end=10.,sampling_rate=30000,plot=False,filter_size=2,sigma=1.,filter=False,probemap=None):
	mm = np.memmap(path, dtype=np.int16, mode='r')
	print(os.path.dirname(path))
	num_channels = get_channel_count(os.path.dirname(path),from_channel_map=False)
	print(num_channels)
	chunk = get_chunk(mm,start,end,num_channels,sampling_rate)
		
	if probemap is not None:
		chunk = chunk[probemap,:]
		plt.imshow(chunk[:,:30000]);plt.gca().set_aspect(100)
		plt.figure()
		
	rms = []
	good_channels = []
	for ch in range(np.shape(chunk)[0]):
		if ch not in skip_channels:
			if filter:
				data = filtr(chunk[ch,:],300,6000,sampling_rate,3)
			else:
				data = chunk[ch,:]
			rms.extend([RMS(data)])
			good_channels.extend([ch])
			
	threshold = np.mean(gaussian_filter1d(rms,filter_size)[::-1][:5])+np.std(gaussian_filter1d(rms,filter_size)[::-1][:5])*sigma		#assumes the last 5 are out of the brain; uses the mean + sd of these 5 as the threshold for pial surface
	
	# print(np.where(np.array(rms)<8.))
	# print(good_channels[np.where(np.array(rms)<8.)[0].astype(int)])
	if plot:
		plt.plot(good_channels,gaussian_filter1d(rms,filter_size))
		plt.gca().axhline(threshold,color='r')
		plt.xlabel('channel number')
		plt.ylabel('spike band RMS')
		#print(np.where(np.array(rms)<6.))
	del mm
	try:
		surface_channel = good_channels[mlab.cross_from_above(gaussian_filter1d(rms,filter_size),threshold)[0]]
		return surface_channel
	except:
		return None

def get_surface_channel_gamma(path,start=2.,end=10.,sampling_rate=2500,plot=False):
	mm = np.memmap(path, dtype=np.int16, mode='r')
	num_channels = get_channel_count(os.path.dirname(path))
	chunk = get_chunk(mm,start,end,num_channels,sampling_rate)
	
	gm = []
	good_channels = []
	for ch in range(np.shape(chunk)[0]):
		if ch not in skip_channels:
			f,pXX = welch_power(chunk[ch,:],start=2,window=8)
			gm.extend([pXX[np.where(f>40.)[0][0]]])
			good_channels.extend([ch])
	threshold = np.max(gm[::-1][:5])	#assumes the last 5 are out of the brain; uses the max gamma on these channels as the threshold
	surface_channel = good_channels[mlab.cross_from_above(gaussian_filter1d(gm,0),threshold)[0]]

	if plot:
		plt.plot(good_channels,gaussian_filter1d(gm,2))
		plt.gca().axhline(threshold,color='r')
	del mm
	return surface_channel

def get_surface_channel_freq(path,frequency_range=[1,5],start=2.,end=10.,sampling_rate=2500,filter_size=2,sigma=2.,plot=False,filter=False,probemap=None):
	mm = np.memmap(path, dtype=np.int16, mode='r')
	num_channels =  get_channel_count(os.path.dirname(path),from_channel_map=False)
	chunk = get_chunk(mm,start,end,num_channels,sampling_rate)
	if probemap is not None:
		chunk = chunk[probemap,:]
	gm = []
	good_channels = []
	for ch in range(np.shape(chunk)[0]):
		if ch not in skip_channels:
			if filter:
				data = filtr(chunk[ch,:],0.1,300,sampling_rate,3)
			else:
				data = chunk[ch,:]
			f,pXX = welch_power(chunk[ch,:],start=2,window=8)
			gm.extend([np.mean(pXX[np.where((f>frequency_range[0])&(f<frequency_range[1]))[0]])])
			good_channels.extend([ch])
	#threshold = np.mean(gm[::-1][:5])	#assumes the last 5 are out of the brain; uses the max gamma on these channels as the threshold
	threshold = np.mean(gaussian_filter1d(gm,filter_size)[::-1][:5])+np.std(gaussian_filter1d(gm,filter_size)[::-1][:5])*sigma

	if plot:
		plt.plot(good_channels,gaussian_filter1d(gm,filter_size))
		plt.gca().axhline(threshold,color='r')
		plt.xlabel('channel number')
		plt.ylabel('power in '+str(frequency_range[0])+' to '+str(frequency_range[1])+' band')
	try:
		surface_channel = good_channels[mlab.cross_from_above(gaussian_filter1d(gm,filter_size),threshold)[-1]]
		return surface_channel
	except:
		return None
	del mm
	return surface_channel

def get_probe_freq(path,frequency_range=[1,5],start=2.,end=10.,sampling_rate=2500.,filter=False,probemap=None):
	mm = np.memmap(path, dtype=np.int16, mode='r')
	num_channels = get_channel_count(os.path.dirname(path),from_channel_map=False)
	chunk = get_chunk(mm,start,end,num_channels,sampling_rate)
	if probemap is not None:
		chunk = chunk[probemap,:]
	gm = []
	good_channels = []
	for ch in range(np.shape(chunk)[0]):
		if ch not in skip_channels:
			if filter != False:
				data = filtr(chunk[ch,:],filter[0],filter[1],sampling_rate,3)
			else:
				data = chunk[ch,:]
			f,pXX = welch_power(chunk[ch,:],start=2,window=8)
			gm.extend([np.mean(pXX[np.where((f>frequency_range[0])&(f<frequency_range[1]))[0]])])
			good_channels.extend([ch])

	del mm
	return gm

def get_probe_spikeband(path,start=2.,end=10.,sampling_rate=30000,plot=False,filter_size=2,sigma=1.,filter=False,probemap=None):
	mm = np.memmap(path, dtype=np.int16, mode='r')
	num_channels = get_channel_count(os.path.dirname(path),from_channel_map=False)
	#print num_channels
	chunk = get_chunk(mm,start,end,num_channels,sampling_rate)
		
	if probemap is not None:
		chunk = chunk[probemap,:]
		plt.imshow(chunk[:,:30000]);plt.gca().set_aspect(100)
		plt.figure()
		
	rms = []
	good_channels = []
	for ch in range(np.shape(chunk)[0]):
		if ch not in skip_channels:
			if filter:
				data = filtr(chunk[ch,:],300,6000,sampling_rate,3)
			else:
				data = chunk[ch,:]
			rms.extend([RMS(data)])
			good_channels.extend([ch])
	del mm
	return rms
#=================================================================================================

#=================================================================================================
#=======plotting=============================================================================
#=================================================================================================
def get_chunk(mm,start,end,channels,sampling_rate=30000):
	chunk = mm[int(start*sampling_rate*int(channels)):int(np.floor(end*sampling_rate*(int(channels))))]
	#print np.shape(chunk)
	return np.reshape(chunk,(int(channels),-1),order='F')  * 0.195

def get_duration(mm,channels,sampling_rate=30000.):
    chunk=np.reshape(mm,(int(channels),-1),order='F')
    return chunk.shape[1]/float(sampling_rate)

def get_spike(spks_path,times,number_channels,pre=0.015,post=0.025,sampling_rate=30000):
    mm = np.memmap(spks_path, dtype=np.int16, mode='r')
    average = np.zeros((number_channels,int(sampling_rate*(pre+post))))
    count=0
    for time in times:
        try:
            temp = traces.get_chunk(mm,time-pre,time+post,number_channels,sampling_rate=sampling_rate)
            average += temp
            count+=1
        except:
            pass
    temp = temp/float(count)
    spikes_average = np.array(average.T - np.mean(average,1).T).T
    del(mm)

def get_lfp(lfp_path,times,number_channels,pre=0.015,post=0.025,sampling_rate=2500):
    mm = np.memmap(lfp_path, dtype=np.int16, mode='r')
    average = np.zeros((number_channels,int(sampling_rate*(pre+post))))
    count=0
    for time in sub_times:
        try:
            temp = traces.get_chunk(mm,time-pre,time+post,number_channels,sampling_rate=sampling_rate)
            average += temp
            count+=1
        except:
            pass
    temp = temp/float(count)
    lfp_average = np.array(average.T - np.mean(average,1).T).T
    del(mm)

def make_range_slider(data,start,window,num_channels=384,channels = [10],sampling_rate=2500,y_spacing=500,CAR=False):
    fig,ax=plt.subplots(figsize=(20,5))
    chunk = get_chunk(data,start,start+window,num_channels,sampling_rate=2500)
    if CAR:
        chunk_CAR = np.mean(chunk, axis=0)
    x = np.linspace(int(start),(start+window),int(window*sampling_rate))
    for i,ch in enumerate(channels):
        chunkch = chunk[ch,:]
        offset = np.mean(chunkch)
        if CAR: chunkch = chunkch - chunk_CAR
        ax.plot(x,chunkch-offset+i*y_spacing,'k',lw=.5)
#=================================================================================================
