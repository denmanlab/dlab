import numpy as np
import scipy, os, glob
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

def recreate_probe_timestamps_from_TTL(directory):
    probe = directory.split('-AP')[0][-1]
    recording_base = os.path.dirname(os.path.dirname(directory))

    with open(os.path.join(recording_base,'sync_messages.txt')) as f:
        lines = f.readlines()
        for line in lines:
            if 'Probe'+probe+'-AP' in line:
                cont_start_sample = int(line.split(':')[1][1:].split('\n')[0])
    f.close()

    TTL_samples = np.load(os.path.join(glob.glob(os.path.join(recording_base,'events')+'/*Probe'+probe+'*AP*')[0],'TTL','sample_numbers.npy'))[::2]
    TTL_timestamps = np.load(os.path.join(glob.glob(os.path.join(recording_base,'events')+'/*Probe'+probe+'*AP*')[0],'TTL','timestamps.npy'))[::2]

    cont_raw = np.memmap(os.path.join(directory,'continuous.dat'),dtype=np.int16)
    cont_samples = np.arange(cont_start_sample, cont_start_sample+(int(cont_raw.shape[0]/384)))
    cont_timestamps = np.zeros(int(cont_raw.shape[0]/384))

    for i,sample in enumerate(TTL_samples):
        ind = sample-cont_start_sample
        cont_samples[ind] = sample
        cont_timestamps[ind] = TTL_timestamps[i]
        if i==0:
            cont_timestamps[:ind]=np.linspace(TTL_timestamps[i]-(1/30000. * len(cont_timestamps[:ind-1]))+1/30000.,TTL_timestamps[i],len(cont_timestamps[:ind]))
            prev_ind =ind
        else:
            cont_timestamps[prev_ind:ind] = np.linspace(cont_timestamps[prev_ind]+1/30000.,TTL_timestamps[i],len(cont_timestamps[prev_ind:ind]))
            prev_ind =ind
    cont_timestamps[ind:]=np.linspace(TTL_timestamps[i]+1/30000.,TTL_timestamps[i]+len(cont_timestamps[ind:])*1/30000.,len(cont_timestamps[ind:]))

    if not os.path.exists(os.path.join(directory,'new_timestamps')):
        os.mkdir(os.path.join(directory,'new_timestamps'))
    np.save(open(os.path.join(directory,'new_timestamps','sample_numbers.npy'),'wb'),cont_samples.astype(np.int64))
    np.save(open(os.path.join(directory,'new_timestamps','timestamps.npy'),'wb'),cont_timestamps.astype(np.float64))

### jordan raw data functions

import seaborn as sns
from open_ephys.analysis import Session

def OE(path):
    'lazy wrappy to return recording session from open ephys'
    session = Session(path)
    recording = session.recordnodes[0].recordings[0]
    return recording

def load_datastream(path, probe, band='ap'):
    '''
    purpose of this is to return a data_stream object. can then access metadata and load raw data from here.
    path: path to recording
    probe: probe name (probeA, probeB, probeC)
    '''
    # normalize case for flexibility
    probe = probe.lower()
    band = band.lower()

    # Get the recording
    recording = OE(path)

    # Search for the correct data stream by examining metadata
    for data_stream in recording.continuous:
        stream_name = data_stream.metadata['stream_name'].lower().replace('-', '_')
        if stream_name == f"{probe}_{band}":
            print(f'confirming stream name: {data_stream.metadata["stream_name"]}')
            return data_stream

    print('No matching bands or probes')
    return None

def get_chunk_OE(path,
              probe, 
            stim_times,
            band = 'ap',
            pre = 100, # time in ms
            post = 500, # time in ms
            chs = np.arange(0,200,1), # channels
            median_subtraction = False,
            ):
    """
    for open ephys data
    Takes in a continuous datastream object (from open ephys) and a list of stimulation times and returns a chunk of the data

    return: data: np.array, shape = (trials, samples, channels)
    """
    
    data_stream = load_datastream(path, probe, band = band)
    sample_rate = data_stream.metadata['sample_rate']
    
    
    pre_samps = int((pre/1000 * sample_rate))
    post_samps = int((post/1000 * sample_rate))
    total_samps = pre_samps + post_samps

    n_chs = len(chs)
    
    response = np.zeros((np.shape(stim_times)[0],total_samps, len(chs)))
    stim_indices = np.searchsorted(data_stream.timestamps, stim_times)
    for i, stim in enumerate(stim_indices):
        start_index = int(stim - ((pre/1000)*sample_rate))
        end_index = int(stim + ((post/1000)*sample_rate))   
        chunk = data_stream.get_samples(start_sample_index = start_index, end_sample_index = end_index, 
                            selected_channels = chs)
        
        if median_subtraction == True:
            corrected_chunk = chunk - np.median(chunk, axis = 0) #subtract offset 
            corrected_chunk = np.subtract(corrected_chunk.T, np.median(corrected_chunk[:,chs[-10]:chs[-1]], axis=1)) #median subtraction

            response[i,:,:] = corrected_chunk.T
        else:
            response[i,:,:] = chunk - np.median(chunk, axis = 0)

    return response

def subtract_offset(data, subtraction_window = None, pre = None, post = None):
    '''
    takes a chunk of data and subtracts the median of the data from each channel

    data: np.array, shape = (trials, samples, channels)
    subtraction_window: 'pre', 'all'. If None, will use full.
    pre: pre window in ms
    post: post window in ms 
    '''
    if subtraction_window == 'pre':
        pre_samps = int((pre/1000 * 30000))
        window = (0, pre_samps)
        pre_data = data[:,window[0]:window[1],:]
        corrected_chunk = data - np.median(pre_data, axis = 1) # the samples axis
    else: # subtract the median of the entire chunk
        corrected_chunk = data - np.median(data, axis = 1)
    
    return corrected_chunk


def median_subtraction(data, channels):
    """median subtraction (not an offset). Like CAR.
    TODO: add ability to select which channels to subtract from

    Args:
        data (np.array): The data to subtract from, shape = (trials, samples, channels).
        channels (list): The channels to subtract from.

    Returns:
        np.array: The data with the median subtracted.
    """

    corrected_data = data.copy()
    corrected_data[:, :, channels] = np.subtract(corrected_data[:, :, channels].T, np.median(corrected_data[:, :, channels], axis=1)).T

    return corrected_data


def find_artifact_start(data_ch, pre, threshold):
    """Find the start of artifact based on a threshold.
    
    Args:
        data_ch: np.array, shape = ch x sample
        threshold: The threshold used to detect the artifact. Function looks for 
                   the first value that exceeds this threshold in absolute terms.
                   
    Returns:
        The index of the first data point that exceeds the threshold.
    """
    # Use the absolute value to find the first occurrence beyond the threshold, 
    # regardless of the sign of the artifact.

    return next((i for i, val in enumerate(np.abs(data_ch)) if val > threshold), int((pre+2)/1000 * 30000))

def align_data(data, pre, post, channels, threshold = 400, median_subtraction = False):
    """ Align the data to the artifact onset based on artifact start times. Optionally perform median subtraction.

    Args:
        data (np.array): The data to align, shape = (trials, samples, channels)
        pre (float): pre-stim time in ms
        post (float): post-stim time in ms
        channels (_type_): number of channels to plot
        median_subtraction (bool): whether to perform median subtraction on the data

    Returns:
        np.array: aligned_data, shape = (trials, total_samps, channels)
    """   
    starts = [find_artifact_start(data[trial, :, 0], pre = pre, threshold = threshold) for trial in range(data.shape[0])] # the sample number of when the artifact first starts 
    # the sample number of when the artifact first starts determined by the find_artifact_start function
    
    pre_samps = (int((pre/1000) * 30000)) 
    post_samps = (int((post/1000) * 30000))
    total_samps = pre_samps + post_samps
    aligned_data = np.zeros((data.shape[0], total_samps, data.shape[2]))
    
    for trial in range(data.shape[0]):
        start = starts[trial]
        chunk = data[trial, start - pre_samps:start + post_samps, :]
        if median_subtraction:
            corrected_chunk = chunk.T - np.median(chunk[:, channels-11:channels-1], axis=1)
            aligned_data[trial, :, :] = corrected_chunk.T
        else:
            aligned_data[trial, :, :] = chunk
    
    return aligned_data




def raw_heatmap(data, pre=1, post=2, dists=None, vmin=None, vmax=None, 
                save=False, save_path=None, save_type='png', title=None, ax=None):
    """
    Plots a heatmap of data with options for customization and saving.

    Args:
        data (np.array): The data to plot, shape = (trials, samples, channels).
        pre (int, optional): pre_window in ms. Defaults to 1.
        post (int, optional): post_window in ms. Defaults to 2.
        dists (np.array, optional): Distance from stimulation array, same shape as channels.
        vmin (float, optional): Minimum value for heatmap scaling. Defaults to None.
        vmax (float, optional): Maximum value for heatmap scaling. Defaults to None.
        save (bool, optional): If True, save the figure. Defaults to False.
        save_path (str, optional): Path to save the figure. Defaults to None.
        save_type (str, optional): Format of saved figure. Defaults to 'png'.
        title (str, optional): Figure title. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Defaults to None (creates new figure).

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """

    # Check if an axes object was provided, if not create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        created_fig = True
    else:
        created_fig = False  # Flag to avoid creating a new figure

    channels = data.shape[2]
    data_to_plot = np.mean(data, axis=0).T  # average across trials

    time_ms = np.linspace(-pre, post, data.shape[1])

    cax = ax.imshow(data_to_plot, aspect='auto', extent=[time_ms[0], time_ms[-1], 0, channels],
                    vmax=vmax, vmin=vmin, origin='lower', cmap='viridis')

    if created_fig:
        fig.colorbar(cax, ax=ax, pad=0.20)
    ax.set_ylabel('Channel')
    ax.set_yticks(np.arange(0, channels, 10))

    if dists is not None:
        ax_dist = ax.twinx()
        ax_dist.set_ylim(ax.get_ylim())
        ax_dist.set_yticks(np.arange(0, len(dists), 25))
        ax_dist.set_yticklabels([int(d) for d in dists[::25]])
        ax_dist.set_ylabel('Distance from Stimulation (units)')
        zero_dist_channel = np.argmin(np.abs(dists))
        ax.axhline(y=zero_dist_channel, color='red', linestyle='--')

    ax.set_xlabel('Time (ms)')
    if title:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()

    if save and created_fig:
        plt.savefig(save_path, format=save_type)

    return ax



def plot_ap(path, probe, stim_times, 
                pre = 5, post = 10, 
                first_ch = 125, last_ch = 175, 
                median_subtraction = False,
            
                spike_overlay = False,
                units = None,
                title = '', 

                n_trials = 10, spacing_mult = 350, 
                save = False, savepath = '', format ='png'):

        '''
        path: recording path, 
        probe: probe (e.g., 'probeA', 'probeB', 'probeC')
        data: 3D array (trials x samples x channels)
        
        stim_times: list of stimulation times
        pre: pre window in ms
        post: post window in ms
        
        first_ch: first channel to plot
        last_ch: last channel to plot

        probeID: probe name ('A', 'B', 'C')
        spike_overlay: whether to overlay spike times from dataframe
        units: dataframe with spike times
        title: title of the plot

        n_trials: number of trials to plot (bc each trial is overlaid)
        spacing_mult: multiplier for spacing between channels
        save: whether to save the plot
        savepath: where to save the plot
        format: format of the saved plot (png, eps, etc)
        '''
        data_stream = load_datastream(path, probe)
        sample_rate = data_stream.metadata['sample_rate']
        data = get_chunk(path, probe, stim_times, 
                             pre = pre, post = post, 
                             chs =np.arange(0,300,1), 
                             median_subtraction = median_subtraction)
        

        probeID = probe.strip('probe')
        total_samps = int((pre/1000 * sample_rate) + (post/1000 * sample_rate))            
        
        if spike_overlay == True:
            stim_indices = np.searchsorted(data_stream.timestamps,stim_times)
            condition = (
                (units['ch'] >= first_ch) &
                (units['ch'] <= last_ch) &
                (units['probe'] == probeID) &
                (units['group'] == 'good')
            )
        
            spikes = np.array(units.loc[condition, 'spike_times'])
            spike_ch = np.array(units.loc[condition, 'ch'])

            spike_dict = {}
            for i, stim in enumerate(stim_indices):
                start_index = int(stim - ((pre/1000)*sample_rate))
                end_index = int(stim + ((post/1000)*sample_rate))  
                window = data_stream.timestamps[start_index:end_index]
                filtered_spikes = [spike_times[(spike_times >= window[0]) & (spike_times <= window[-1])] for spike_times in spikes]  
                spike_dict[i] = filtered_spikes

        ## plotting 
        
        trial_subset = np.linspace(0,len(stim_times)-1, n_trials) #choostrse random subset of trials to plot 
        trial_subset = trial_subset.astype(int)
        #set color maps
        cmap = sns.color_palette("crest",n_colors = n_trials)
        #cmap = sns.cubehelix_palette(n_trials)
        colors = cmap.as_hex()
        if spike_overlay == True:
            cmap2 = sns.color_palette("ch:s=.25,rot=-.25", n_colors = len(spikes))
            colors2 = cmap2.as_hex()
        fig=plt.figure(figsize=(16,24))
        time_window = np.linspace(-pre,post,(total_samps))
        for trial,color in zip(trial_subset,colors):
        
            for ch in range(first_ch,last_ch): 
                plt.plot(time_window,data[trial,:,ch]+ch*spacing_mult,color=color)
        
            if spike_overlay == True:
                for i,ch in enumerate(spike_ch): 

                    if spike_dict[trial][i].size > 0:
                        for spike in spike_dict[trial][i]:
                            spike = spike - stim_times[trial]
                            plt.scatter(spike*1000, (spike/spike) + ch*spacing_mult, 
                            alpha = 0.75, color = colors2[i], s = 500)
        
        plt.gca().axvline(0,ls='--',color='r')       
        plt.xlabel('time from stimulus onset (ms)')
        plt.ylabel('uV')
        plt.title(title)
        
        if save == True:
            plt.gcf().savefig(savepath,format=format,dpi=600)