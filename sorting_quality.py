##imports===============================================================================
from scipy.spatial.distance import cdist
import numpy as np
import os, sys,glob, copy, csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from  dlab.continuous_traces import get_channel_count, filtr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from scipy.cluster.vq import kmeans2
##===============================================================================





##===============================================================================
#probe geometry, for the summary figure only
option234_xpositions = np.zeros((192,2))
option234_ypositions = np.zeros((192,2))
option234_positions = np.zeros((383,2))
option234_positions[:,0][::4] = 21
option234_positions[:,0][1::4] = 53
option234_positions[:,0][2::4] = 5
option234_positions[:,0][3::4] = 37
option234_positions[:,1] = np.floor(np.linspace(383,0,383)/2) * 20
##===============================================================================

##helper functions===============================================================================
def read_kilosort_params(filename):
    f=open(filename)
    params = {}
    for line in list(f):
        try:
            params[line.split(' =')[0]]=line.split('= ')[1].replace('\r\n','')
        except:
            pass
    return params

def read_cluster_groups_CSV(directory):
    cluster_id = [];
    if os.path.isfile(os.path.join(directory,'cluster_group.tsv')):
        cluster_id = [row for row in csv.reader(open(os.path.join(directory,'cluster_group.tsv')))];
    else:
        if os.path.isfile(os.path.join(directory,'cluster_groups.csv')):
            cluster_id = [row for row in csv.reader(open(os.path.join(directory,'cluster_groups.csv')))];
        else:
            print('could not find cluster groups csv or tsv')
            return None
    good=[];mua=[];unsorted=[]
    for i in np.arange(1,np.shape(cluster_id)[0]):
        if cluster_id[i][0].split('\t')[1] == 'good':#if it is a 'good' cluster by manual sort
            good.append(cluster_id[i][0].split('\t')[0])
        if cluster_id[i][0].split('\t')[1] == 'mua':#if it is a 'good' cluster by manual sort
            mua.append(cluster_id[i][0].split('\t')[0])
        if cluster_id[i][0].split('\t')[1] == 'unsorted':#if it is a 'good' cluster by manual sort
            unsorted.append(cluster_id[i][0].split('\t')[0])
    return (np.array(good).astype(int),np.array(mua).astype(int),np.array(unsorted).astype(int))
    
def count_unique(x):
    values=[]
    instances=[]
    for v in np.unique(x):
        values.extend([v])
        instances.extend([len(np.where(np.array(x)==v)[0].flatten())])
    return values, instances

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" 

def load_waveforms(data,channel,times,pre=0.5,post=1.5,channels=384,sampling_rate=30000):
    #input can be memory mapped file or a string specifiying the file to memory map. if string, deletes the memory mapped object when done, for hygiene.
    pre = pre * .001
    post = post * .001
    channel = int(channel)
    channels = int(channels)
    if type(data)==str:
        mm = np.memmap(data, dtype=np.int16, mode='r')
    else:
        mm=data
    waveforms=[]
    for i in times:
        start = int((i - pre) * sampling_rate) * int(channels)
        temp = mm[start:start+int((pre+post)*sampling_rate*channels)][channel::channels]# - mm[start:start+int((pre+post)*sampling_rate*channels)][channel::channels][0]
        temp = temp - temp[0]
        waveforms.extend([temp * 0.195])
    if type(data)==str:
        del mm
    return waveforms

def mean_waveform(rawdata,times,pre=0.5,post=1.5,channels=384,sampling_rate=30000):
    mean_waveform = []#np.zeros(channels,int((pre+post)*.001)*sampling_rate*channels)
    for i,ch in enumerate(np.linspace(0,channels-1,channels).astype(int)):
        try:
            w = load_waveforms(rawdata,ch,times,pre,post,channels,sampling_rate)
        except:
            w = np.zeros((len(times),int(((pre+post)/1000.)*sampling_rate)))
        mean_waveform.append(np.mean(w,axis=0))#[i,:]=np.mean(w).flatten()
    return mean_waveform

def probe_waveforms(rawdata,times,pre=0.5,post=1.5,channels=384,sampling_rate=30000):
    probe_waveform = np.zeros((channels,int((pre+post)*.001*sampling_rate),len(times)))
    for i,ch in enumerate(np.linspace(0,channels-1,channels).astype(int)):
        try:
            probe_waveform[i,:,:] = np.array(load_waveforms(rawdata,ch,times,pre,post,channels,sampling_rate)).T
            probe_waveform[i,:,:] -= np.mean(probe_waveform[i,:,:])
        except:
            probe_waveform[i,:,:] = 0.#np.zeros((len(times),int(((pre+post)/1000.)*sampling_rate)))
    return probe_waveform.T

def load_phy_template(path,site_positions = option234_positions,**kwargs):
# load spike data that has been manually sorted with the phy-template GUI
# the site_positions should contain coordinates of the channels in probe space. for example, in um on the face of the probe
# returns a dictionary of 'good' units, each of which includes:
#	times: spike times, in seconds
#	template: template used for matching
#	ypos: y position on the probe, calculated from the template. requires an accurate site_positions. averages template from 100 spikes.
#	xpos: x position on the probe, calcualted from the template. requires an accurate site_positions. averages template from 100 spikes.
    clusters = np.load(open(os.path.join(path,'spike_clusters.npy'),'rb'))
    spikes = np.load(open(os.path.join(path,'spike_times.npy'),'rb'))
    spike_templates = np.load(open(os.path.join(path,'spike_templates.npy'),'rb'))
    templates = np.load(open(os.path.join(path,'templates.npy'),'rb'))
    cluster_id = [];
    #[cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_group.tsv')))];
    if os.path.isfile(os.path.join(path,'cluster_group.tsv')):
        cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_group.tsv')))];
    else:
        if os.path.isfile(os.path.join(path,'cluster_groups.csv')):
            cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))];
        else:
            print('could not find cluster groups csv or tsv')
            return None
    if 'sampling_rate' in kwargs.keys():
        samplingrate = kwargs['sampling_rate']
    else:
        samplingrate =30000.
        print('no sampling rate specified, using default of 30kHz')

    units = {}
    for i in np.arange(1,np.shape(cluster_id)[0]):
        if cluster_id[i][0].split('\t')[1] != 'garabge' :#:or cluster_id[i][0].split('\t')[1] == 'unsorted' :#if it is a 'good' cluster by manual sort

            unit = int(cluster_id[i][0].split('\t')[0])
            units[str(unit)] = {}

            #get the unit spike times
            units[str(unit)]['times'] = spikes[np.where(clusters==unit)]/samplingrate
            units[str(unit)]['times'] = units[str(unit)]['times'].flatten()

            #get the mean template used for this unit
            all_templates = spike_templates[np.where(clusters==unit)].flatten()
            n_templates_to_subsample = 100
            random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
            mean_template = np.mean(random_subsample_of_templates,axis=0)
            units[str(unit)]['template'] = mean_template

            #take a weighted average of the site_positions, where the weights is the absolute value of the template for that channel
            #this gets us the x and y positions of the unit on the probe.
            weights = np.zeros(site_positions.shape)
            for channel in range(site_positions.shape[0]):
                weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
            weights = weights/np.max(weights)
            (xpos,ypos)=np.average(site_positions,axis=0,weights=weights)
            units[str(unit)]['waveform_weights'] = weights
            units[str(unit)]['xpos'] = xpos
            units[str(unit)]['ypos'] = ypos - site_positions[-1][1]

    return units    
##===============================================================================    



##===============================================================================
def ISIviolations(spikeTrain, refDur, minISI):
    #modified from cortex-lab/sortingQuality GitHub by nick s. 
    isis = np.diff(spikeTrain)
    nSpikes = len(spikeTrain)
    numViolations = sum(isis<refDur) 
    violationTime = 2*nSpikes*(refDur-minISI)
    totalRate = nSpikes/(spikeTrain[-1] - spikeTrain[0])
    violationRate = numViolations/violationTime
    fpRate = violationRate/totalRate
    if fpRate > 1.:
        fpRate = 1. # it is nonsense to have a rate > 1; a rate > 1 means the assumputions of this analysis are failing
    return fpRate, numViolations

def isiViolations(directory,time_limits=None,tr=.0015,tc=0):
    spike_clusters_path = os.path.join(directory,'spike_clusters.npy')
    spike_templates_path = os.path.join(directory,'spike_templates.npy')
    spike_times_path = os.path.join(directory,'spike_times.npy')
    params_path = os.path.join(directory,'params.py')
    
    if time_limits == None:
        time_limits=[0,1e7]

    print(' ')
    print('loading data for ISI computation...')
    if os.path.isfile(spike_clusters_path):
        spike_clusters = np.load(spike_clusters_path)
    else:
        spike_clusters = np.load(spike_templates_path)

    params = read_kilosort_params(params_path)
    spike_times = np.load(spike_times_path) / float(params['sample_rate'])

    print('computing ISI violations...')
    cluster_IDs = np.unique(spike_clusters)
    isiV = np.zeros(np.shape(cluster_IDs)[0])
    for i,cluster_ID in enumerate(cluster_IDs):
        all_spikes = spike_times[np.where(spike_clusters==cluster_ID)[0]].flatten()
        if all_spikes[-1] < time_limits[0] or all_spikes[0] > time_limits[1]:
            isiV[i] = np.nan
        else:
            spikes = all_spikes[np.where(all_spikes > time_limits[0])[0][0]:np.where(all_spikes < time_limits[1])[0][-1]]
            if len(spikes) > 1:
                (fp_rate, num_violations) = ISIviolations(spikes,tr,tc)
                #print fp_rate
                isiV[i] = fp_rate
                n_spikes = len(spikes)
                print('\rcluster '+str(cluster_ID)+': '+str(num_violations)+' violations ('+str(n_spikes)+' spikes), '+str(fp_rate)+' estimated FP rate',)
            else:
                isiV[i] = np.nan
    
    return cluster_IDs,isiV
    
def cluster_signalToNoise(directory,time_limits=None,filename='experiment1_100-0_0.dat',sigma=5.,number_to_average=250,plots=False,channels='all'):
    spike_clusters_path = os.path.join(directory,'spike_clusters.npy')
    spike_templates_path = os.path.join(directory,'spike_templates.npy')
    spike_times_path = os.path.join(directory,'spike_times.npy')
    params_path = os.path.join(directory,'params.py')
    
    if time_limits == None:
        time_limits=[0,1e7]

    print(' ')
    print('loading data for s/n computation...')
    if os.path.isfile(spike_clusters_path):
        spike_clusters = np.load(spike_clusters_path)
    else:
        spike_clusters = np.load(spike_templates_path)
    try:
        rawdata = np.memmap(os.path.join(directory,filename), dtype=np.int16, mode='r')
    except:
        print('could not load spike data. is the filename correct? (default: experiment1_100-0_0.dat)')
        return None
    spike_templates = np.load(spike_templates_path)
    templates = np.load(os.path.join(directory,'templates.npy'))
    
    params = read_kilosort_params(params_path)
    spike_times = np.load(spike_times_path) / float(params['sample_rate'])
    if channels == 'all':
        channels = get_channel_count(directory)
    
    site_positions=option234_positions[np.linspace(0,channels-1,channels).astype(int)]
    data = load_phy_template(directory,site_positions)
   
    print('computing Quian Quiroga signal/noise ratio...')
    cluster_IDs = np.unique(spike_clusters)
    sn_peak = np.zeros(np.shape(cluster_IDs)[0])
    sn_mean = np.zeros(np.shape(cluster_IDs)[0])
    #spike_amplitudes = []
    sn_all=[]
    for i,cluster_ID in enumerate(cluster_IDs):
        print('\rcluster '+str(cluster_ID)+': '+str(i)+'/'+str(len(cluster_IDs)),)
        #peak_y_channel = np.where(data[clusterID]['waveform_weights'] == np.max(data[clusterID]['waveform_weights']))[0][0]
        all_spikes = spike_times[np.where(spike_clusters==cluster_ID)[0]].flatten()
        if all_spikes[-1] < time_limits[0] or all_spikes[0] > time_limits[1] or str(cluster_ID) not in data.keys():
            sn_peak[i] = np.nan
            sn_mean[i] = np.nan
            sn_all.append([np.nan])
        else:
            spikes = all_spikes[np.where(all_spikes > time_limits[0])[0][0]:np.where(all_spikes < time_limits[1])[0][-1]]
            channels_with_template = [ch for ch,t in enumerate(data[str(cluster_ID)]['waveform_weights']) if np.max(np.abs(t))>0.1] # <----- get the channels that have a template > 20uV
            if len(spikes) > number_to_average:
                sub_times = np.random.choice(spikes,number_to_average,replace=False)
            else:
                sub_times = spikes
            random_times = np.random.rand(number_to_average) * (np.max(spike_times)-np.min(spike_times)) + np.min(spike_times)
            s_ns = []
            if plots:
                plt.figure()
            for channel in channels_with_template:
                ws = load_waveforms(os.path.join(directory,filename),channel,sub_times,channels=channels)
                rs = load_waveforms(os.path.join(directory,filename),channel,random_times,channels=channels)
                
                mean = np.mean(ws,axis=0)       

                # if time_limits[1]==1e7:
                #     cunk = rawdata[int(650*int(params['sample_rate'].strip('.'))*channels):int(660*int(params['sample_rate'].strip('.'))*channels)][channel::channels] 
                # else:
                #     cunk = rawdata[int(time_limits[0]*int(params['sample_rate'].strip('.'))*channels):int(time_limits[1]*int(params['sample_rate'].strip('.'))*channels)][channel::channels] 
                # channel_chunk = filtr(cunk,300,6000,float(params['sample_rate']),3) * 0.195 # <-- re-filter and put into uV
                
                #noise = sigma * np.median(np.abs(channel_chunk)/0.6725)
                noise = sigma * np.median(np.abs(np.array(rs))/0.6725) # <--- as defined by Quian Quiroga 2004
                signal = np.max(np.abs(mean)) # <-- convert to uV
                # if signal/noise > np.max(s_ns):
                #     amplitudes = [np.max(np.abs(w)) for w in np.array(waveforms).T]
                s_ns.extend([signal/noise])
                if plots:
                    plt.plot(mean)
            if plots:
                plt.title(str(np.max(s_ns)))
    
            sn_peak[i] = np.max(s_ns)
            sn_mean[i] = np.mean(s_ns)
            sn_all.append(s_ns)
            #spike_amplitudes.append(amplitudes)
            print('\rcluster '+str(cluster_ID)+': '+str(np.max(s_ns))+' sn',)
    return cluster_IDs, sn_peak, sn_mean, sn_all# spike_amplitudes,sn_all

def masked_cluster_quality(directory,time_limits=None,n_fet=3,minimum_number_of_spikes=10):
    pc_features_path = os.path.join(directory,'pc_features.npy')
    pc_features_ind_path = os.path.join(directory,'pc_feature_ind.npy')
    spike_clusters_path = os.path.join(directory,'spike_clusters.npy')
    spike_templates_path = os.path.join(directory,'spike_templates.npy')
    spike_times_path = os.path.join(directory,'spike_times.npy')
    params_path = os.path.join(directory,'params.py')
    params = read_kilosort_params(params_path)
    
    try:
        pc_features = np.load(pc_features_path)
    except:
        print('loading PC features failed.')
        return None
    pc_feature_ind = np.load(pc_features_ind_path)

    if os.path.isfile(spike_clusters_path):
        print('building features matrix from clusters / templates')
        spike_clusters = np.load(spike_clusters_path)
        spike_templates = np.load(spike_templates_path)
        spike_times = np.load(spike_times_path) / float(params['sample_rate'])
        
        cluster_IDs = np.unique(spike_clusters)
        n_clusters = len(cluster_IDs)
        n_spikes = len(spike_clusters)
        n_fet_per_chan = np.shape(pc_features)[1]
        n_templates = np.shape(pc_feature_ind)[0]
        new_fet = np.zeros((n_spikes,n_fet_per_chan,n_fet))
        new_fet_inds = np.zeros((n_clusters,n_fet))
        
        if time_limits == None:
            time_limits=[0,1e7]

    else:
        print('spike_clusters do not exist, using spike_templates instead')
    
    print('computing cluster qualities...')
    (cluster_IDs,unit_quality,contamination_rate,flda) = masked_cluster_quality_sparse(spike_clusters,pc_features,pc_feature_ind,spike_times,spike_templates,time_limits)
    return cluster_IDs,unit_quality,contamination_rate, flda


def masked_cluster_quality_sparse(spike_clusters,pc_features,pc_feature_ind,spike_times,spike_templates,time_limits=None,n_fet=3,fet_n_chans=6):
    fet_N = np.shape(pc_features)[1] * fet_n_chans
    N = len(spike_clusters)
    cluster_IDs = np.unique(spike_clusters)
    unit_quality = np.zeros(len(cluster_IDs))
    contamination_rate = np.zeros(len(cluster_IDs))
    flda = np.zeros(len(cluster_IDs))
    print('number of clusters: '+str(np.shape(unit_quality)[0]))
    if time_limits == None:
        time_limits=[0,1e7]
        
    for i,cluster_ID in enumerate(cluster_IDs):
        all_spikes = spike_times[np.where(spike_clusters==cluster_ID)[0]].flatten()
        if all_spikes[-1] < time_limits[0] or all_spikes[0] > time_limits[1]:
            unit_quality[i] = 0;
            contamination_rate[i] =np.nan;
        else:
            these_spikes = np.where(spike_clusters==cluster_ID)[0][np.where(all_spikes > time_limits[0])[0][0]:np.where(all_spikes < time_limits[1])[0][-1]+1]
            n_spikes_in_cluster = len(these_spikes)
            if n_spikes_in_cluster < fet_n_chans or n_spikes_in_cluster > N/2.:
                unit_quality[i] = 0;
                contamination_rate[i] =np.nan;
            else:
                fet_this_cluster = pc_features[these_spikes,:n_fet,:fet_n_chans]#.reshape((len(these_spikes),-1))
                #print 'this cluster PCs: '+str(np.shape(fet_this_cluster))
                
                these_templates = spike_templates[these_spikes]
                #count the templates in this unit and their frequency of occurence:
                (included_templates,instances) = count_unique(these_templates)
                #use the template that occurs most frequently:
                this_template = included_templates[np.where(instances==np.max(instances))[0][0]]
                
                this_cluster_chans = pc_feature_ind[this_template,:fet_n_chans]
                other_clusters_IDs = []
                fet_other_clusters = []
                for ii,cluster_2 in enumerate(cluster_IDs):
                    try:
                        if cluster_2 != cluster_ID:
                            all_spikes_2 = spike_times[np.where(spike_clusters==cluster_2)[0]].flatten()
                            these_spikes_2 = np.where(spike_clusters==cluster_2)[0][np.where(all_spikes_2 > time_limits[0])[0][0]:np.where(all_spikes_2 < time_limits[1])[0][-1]+1]
                            these_templates_2 = spike_templates[these_spikes_2]
                            #count the templates in this unit and their frequency of occurence:
                            (included_templates_2,instances_2) = count_unique(these_templates_2)
                            #use the template that occurs most frequently:
                            this_template_2 = included_templates_2[np.where(instances_2==np.max(instances_2))[0][0]]
                            #get the channels that have signal in this template
                            cluster_2_chans = pc_feature_ind[this_template_2]
                                    
                            if np.any(np.in1d(this_cluster_chans,cluster_2_chans)):  #<---continue only if the second cluster has an overlapping channel with the main cluster  
                                #spikes_other_clusters.extend(these_spikes_2)
                                fet_cluster2 = np.zeros((np.shape(these_spikes_2)[0],n_fet,fet_n_chans)) #<---make an empty matrix with dimensions: [number of spikes,number of features, number of channels]
                                #get the pca values, for each channel and each feature, that overlap with the main cluster in question
                                for iii,ch in enumerate(this_cluster_chans):
                                    if np.in1d(this_cluster_chans,cluster_2_chans)[iii]:
                                        fet_cluster2[:,:,iii] = pc_features[these_spikes_2,:n_fet,np.where(ch==cluster_2_chans)[0][0]]
                                fet_other_clusters.extend(fet_cluster2)
                    except:
                        pass
                            
                #reshape to the arrays to dimensions: [number of spikes, (number of features * number of channels)]
                try:
                    fet_other_clusters = np.array(fet_other_clusters).reshape((np.shape(fet_other_clusters)[0],-1))
                    fet_this_cluster = fet_this_cluster.reshape((len(these_spikes),-1))
                except ValueError:
                    pass

                #pass the features to core to calculate isolation distance and mahalanobis contamination
                unit_quality[i],contamination_rate[i] = masked_cluster_quality_core(fet_this_cluster,fet_other_clusters,plots=False)
                
                #pass the features to flda to calculate the d prime of this cluster from all other spikes, using Fisher's LDA
                try:
                    flda[i] = masked_cluster_quality_flda(fet_this_cluster,fet_other_clusters,plots=False)
                except ValueError:
                    pass
                #pass spike times to neighbor similarity to measure the distance between 
                #neighbor_similarity[i],spike_similarity = masked_waveform_similarity(these_spikes,spikes_other_clusters,number_to_average=250,metric='euclidean')
                
                #plans for the future:
                #l_ratio[i] = masked_cluster_quality_lratio(fet_this_cluster,fet_other_clusters,plots=False)

            
        print('\rcluster '+str(cluster_ID)+': # spikes:'+str(np.shape(all_spikes)[0])+'  iso. distance:'+str(unit_quality[i])+'  contamination:'+str(contamination_rate[i])+'  '+str(i+1)+'/'+str(np.shape(unit_quality)[0]),)
    return cluster_IDs,unit_quality,contamination_rate,flda

# def masked_waveform_similarity(directory,number_to_average=250,metric='euclidean'):
#     params_path = os.path.join(directory,'params.py')
#     filename = read_kilosort_params(params_path)['dat_path'][1:-1]
#     
#     if len(spikes) > number_to_average:
#         sub_times = np.random.choice(spikes,number_to_average,replace=False)
#     else:
#         sub_times = spikes
#         
#     random_times = np.random.rand(number_to_average) * (np.max(spike_times)-np.min(spike_times)) + np.min(spike_times)
# 
#     for channel in channels_with_template:
#         ws = load_waveforms(os.path.join(directory,filename),channel,sub_times,channels=channels)
#         
#     return neighbor_similarity_dprime,spike_by_spike_similarity

def masked_cluster_quality_flda(fet_this_cluster,fet_other_clusters,plots=False):
    #based on documentation here: http://sebastianraschka.com/Articles/2014_python_lda.html
    flda =LDA(n_components=1)
    X = np.concatenate((fet_this_cluster,
                        fet_other_clusters))
    y = np.concatenate((np.zeros(np.shape(fet_this_cluster)[0]),
                        np.ones(np.shape(fet_other_clusters)[0])))
    X_flda = flda.fit_transform(X, y)
    
    flda_this_cluster  = X_flda[:np.shape(fet_this_cluster)[0]]
    flda_other_cluster = X_flda[np.shape(fet_this_cluster)[0]:]
    
    dprime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster))/np.sqrt(0.5*(np.std(flda_this_cluster)**2+np.std(flda_other_cluster)**2))
    return dprime

def masked_cluster_quality_core(fet_this_cluster,fet_other_clusters,point_limit=20000000,plots=False):
    n = np.shape(fet_this_cluster)[0]
    n_other = np.shape(fet_other_clusters)[0]
    n_fet = np.shape(fet_this_cluster)[1]

    if n_other > n and n > n_fet:
        if n > point_limit:
            random_indices = np.random.choice(n,point_limit,replace=False)
            fet_this_cluster = fet_this_cluster[random_indices,:]
        if n_other > point_limit:
            random_indices = np.random.choice(n_other,point_limit,replace=False)
            fet_other_clusters = fet_other_clusters[random_indices,:]

        md = np.sort(cdist(fet_this_cluster.mean(0).reshape(1,fet_this_cluster.shape[1]),
                            fet_other_clusters,
                            'mahalanobis')[0])
        md_self = np.sort(cdist(fet_this_cluster.mean(0).reshape(1,fet_this_cluster.shape[1]),
                                 fet_this_cluster,
                                 'mahalanobis')[0])
        #print fet_this_cluster.mean(0).reshape(1,fet_this_cluster.shape[1])
        if plots:
            plt.figure()
            plt.plot(fet_this_cluster[:,0],fet_this_cluster[:,1],'r.')
            plt.plot(fet_other_clusters[:n,0],fet_other_clusters[:n,1],'b.')
            plt.plot(fet_this_cluster.mean(0).reshape(1,fet_this_cluster.shape[1])[0][0],fet_this_cluster.mean(0).reshape(1,fet_this_cluster.shape[1])[0][1],'*',color='#ffcccc',ms=12)
            #
            plt.figure()
            plt.hist(md[:n],bins=100,range=(0,10),color='b')
            plt.hist(md_self,bins=100,range=(0,10),color='r')
            plt.title('iso: '+str(np.max(md[:n])))
        unit_quality = np.max(md[:n])
        contamination_rate = 1 - (tipping_point(md_self,md) / float(len(md_self)))
    else:
        unit_quality = 0
        contamination_rate = np.nan
    return unit_quality, contamination_rate



    
def tipping_point(x,y):
# Input: x, y  are sorted ascending arrays of positive numbers
# Output: minimal pos s.t. sum(x > x(pos)) <= sum(y < x(pos))

#original:
# algorithm here is to sort x and y together, and determine the indices of
# x in this sorted list (call this xInds). Then, xInds-(1:length(xInds))
# will be the number of y's that are less than that value of x.

# translated from matlab (by nick steinmetz) to python:
# algorithm here is to sort x and y together, and determine how many y are less than x 
# in the first len(x) instances of the sorted together array.


    #pos = [ind for ind,together in enumerate(np.sort(np.array(zip(x,y)).flatten())) if np.any(together==x)][:len(x)][-1]-len(x)
    #pos = np.where(np.array(np.sort(np.concatenate((x,y)).flatten()))[:len(x)] == np.sort(x)[-1])[0][0] - len(x)
    #print str(np.where(np.array(np.sort(np.array(zip(x,y)).flatten()))== np.sort(x)[-1])[0][0])+' furthest self: '+str(x[-1])+'    closest other:'+str(y[0])
    
    pos =  sum(np.in1d(np.array(np.sort(np.concatenate((x,y)).flatten()))[:len(x)],np.sort(x)))

    return pos






##===============================================================================
##plotting functions
##===============================================================================

def placeAxesOnGrid(fig,dim=[1,1],xspan=[0,1],yspan=[0,1],wspace=None,hspace=None,):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec
    
    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively
        
    returns:
        subaxes handles
        
    written by doug ollerenshaw
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100,100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(dim[0],dim[1],
                                                  subplot_spec=outer_grid[int(100*yspan[0]):int(100*yspan[1]),int(100*xspan[0]):int(100*xspan[1])],
                                                  wspace=wspace, hspace=hspace)
    

    #NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0]*[dim[1]*[fig]] #filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            inner_ax[row][col] = plt.Subplot(fig, inner_grid[idx])
            fig.add_subplot(inner_ax[row,col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist() #remove redundant dimension
    return inner_ax
def cleanAxes(ax,bottomLabels=False,leftLabels=False,rightLabels=False,topLabels=False,total=False):
    ax.tick_params(axis='both',labelsize=10)
    ax.spines['top'].set_visible(False);
    ax.yaxis.set_ticks_position('left');
    ax.spines['right'].set_visible(False);
    ax.xaxis.set_ticks_position('bottom')
    if not bottomLabels or topLabels:
        ax.set_xticklabels([])
    if not leftLabels or rightLabels:
        ax.set_yticklabels([])
    if rightLabels:
        ax.spines['right'].set_visible(True);
        ax.spines['left'].set_visible(False);
        ax.yaxis.set_ticks_position('right');
    if total:
        ax.set_frame_on(False);
        ax.set_xticklabels('',visible=False);
        ax.set_xticks([]);
        ax.set_yticklabels('',visible=False);
        ax.set_yticks([])
def psth_line(times,triggers,pre=0.5,timeDomain=False,post=1,binsize=0.05,ymax=75,yoffset=0,output='fig',name='',color='#00cc00',linewidth=0.5,axes=None,labels=True,sparse=False,labelsize=18,axis_labelsize=20,error='',alpha=0.5,**kwargs):
    post = post + 1
    peris=[]#np.zeros(len(triggers),len(times))
    p=[]
    if timeDomain:
        samplingRate = 1.0
    else:
        samplingRate = samplingRate
        
    times = np.array(times).astype(float) / samplingRate + pre
    triggers = np.array(triggers).astype(float) / samplingRate

    numbins = (post+pre) / binsize 
    bytrial = np.zeros((len(triggers),numbins))
    for i,t in enumerate(triggers):
        
        if len(np.where(times >= t - pre)[0]) > 0 and len(np.where(times >= t + post)[0]) > 0:
            start = np.where(times >= t - pre)[0][0]
            end = np.where(times >= t + post)[0][0]
            for trial_spike in times[start:end-1]:
                if float(trial_spike-t)/float(binsize) < float(numbins):
                    bytrial[i][(trial_spike-t)/binsize-1] +=1   
        else:
        	 pass
             #bytrial[i][:]=0
        #print 'start: ' + str(start)+'   end: ' + str(end)

    variance = np.std(bytrial,axis=0)/binsize/np.sqrt((len(triggers)))
    hist = np.mean(bytrial,axis=0)/binsize
    edges = np.linspace(-pre,post,numbins)

    if output == 'fig':
        if error == 'shaded':
            if 'shade_color' in kwargs.keys():
                shade_color=kwargs['shade_color']
            else:
                shade_color=color    
            if axes == None:
                plt.figure()
                axes=plt.gca()
            plt.locator_params(axis='y',nbins=4)
            upper = hist+variance
            lower = hist-variance
            axes.fill_between(edges[2:-1],upper[2:-1]+yoffset,hist[2:-1]+yoffset,alpha=alpha,color='white',facecolor=shade_color)
            axes.fill_between(edges[2:-1],hist[2:-1]+yoffset,lower[2:-1]+yoffset,alpha=alpha,color='white',facecolor=shade_color)
            axes.plot(edges[2:-1],hist[2:-1]+yoffset,color=color,linewidth=linewidth)
            axes.set_xlim(-pre,post-1)
            axes.set_ylim(0,ymax);
            if sparse:
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            else:
                if labels:
                    axes.set_xlabel(r'$time \/ [s]$',fontsize=axis_labelsize)
                    axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=axis_labelsize)
                    axes.tick_params(axis='both',labelsize=labelsize)
            axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
            axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')   
            axes.set_title(name,y=0.5)
            return axes 
        else:
            if axes == None:
                plt.figure()
                axes=plt.gca()
            f=axes.errorbar(edges,hist,yerr=variance,color=color)
            axes.set_xlim(-pre,post - 1)
            if ymax=='auto':
                pass
            else:
                axes.set_ylim(0,ymax)
            if sparse:
                axes.set_xticklabels([])
                axes.set_yticklabels([])
            else:
                if labels:
                    axes.set_xlabel(r'$time \/ [s]$',fontsize=axis_labelsize)
                    axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=axis_labelsize)
                    axes.tick_params(axis='both',labelsize=labelsize)
            axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
            axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')   
            axes.set_title(name)
            return axes
    if output == 'hist':
        return (hist,edges)    
    if output == 'p':
        return (edges,hist,variance)
        

def project_linear_quality(data,labels,plots=False):
    flda =LDA(n_components=1)
    if np.shape(data)[0] == np.shape(labels)[0]:
        X_flda = flda.fit_transform(data, labels)
        return X_flda
    else:
        print('dimensions of data do not match labels')
        return None

def get_spike_amplitudes(datapath,channel,times,pre=0.5,post=1.5,channels=384,sampling_rate=30000):
    pre = pre * .001
    post = post * .001
    mm = np.memmap(datapath, dtype=np.int16, mode='r')
    amplitudes = []
    channel = int(channel)
    for i in times:
        start = int((i - pre) * sampling_rate) * int(channels)
        temp=mm[start:start+int((pre+post)*sampling_rate*channels)][channel::channels]
        amplitudes.extend([np.max(np.abs(temp-temp[0]))])
    return amplitudes

def get_PCs(datapath,channel,times,PC=1,pre=0.5,post=1.5,channels=384,sampling_rate=30000):
    pre = pre * .001
    post = post * .001
    mm = np.memmap(datapath, dtype=np.int16, mode='r')
    PCs = []
    for i in times:
        start = int((i - pre) * sampling_rate) * channels
        amplitudes.extend([np.max(np.abs(mm[start:start+(pre+post)*sampling_rate*channels][channel::channels] - mm[start:start+(pre+post)*sampling_rate*channels][channel::channels][0]))])
    return PCs

# %%

def neuron_fig(clusterID,df,sortpath,filename='experiment1_102-0_0.dat',time_limits=None,timeplot_binsize=60.,neighbor_colors=["#67572e","#50a874","#ff4d4d"]):

    channels = 175 #get_channel_count(sortpath)
    site_positions=option234_positions[np.linspace(0,channels-1,channels).astype(int)]
    data = load_phy_template(sortpath,site_positions)
    
    cluster_ID = int(clusterID)
    cluster_IDs = np.load(os.path.join(sortpath,'spike_clusters.npy'))
    pc_data = np.load(os.path.join(sortpath,'pc_features.npy'))
    pc_ind_data = np.load(os.path.join(sortpath,'pc_feature_ind.npy'))
    params_path = os.path.join(sortpath,'params.py')
    params = read_kilosort_params(params_path)
    spike_times_data = np.load(os.path.join(sortpath,'spike_times.npy'))/ float(params['sample_rate'])
    spike_templates = np.load(os.path.join(sortpath,'spike_templates.npy'))
    
    datapath=os.path.join(sortpath,filename)
    #df_line = df[df.clusterID==int(clusterID)]
    
    fig = plt.figure(figsize=(11,8.5))
    ax_text = placeAxesOnGrid(fig,xspan=[0.0,0.4],yspan=[0,0.1])
    ax_position = placeAxesOnGrid(fig,xspan=[0,0.1],yspan=[0.12,1.0])
    ax_waveform = placeAxesOnGrid(fig,xspan=[0.2,0.45],yspan=[0.12,0.65])
    ax_time = placeAxesOnGrid(fig,xspan=[0.2,1.0],yspan=[0.82,1.0])
    ax_PCs = placeAxesOnGrid(fig,xspan=[0.5,0.8],yspan=[0,0.35])
    ax_ACG = placeAxesOnGrid(fig,dim=[1,2],xspan=[0.55,1.0],yspan=[0.5,0.7])
    #ax_neighbor_waveform_1 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0,0.13])
    #ax_neighbor_waveform_2 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0.13,0.26])
    #ax_neighbor_waveform_3 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0.26,0.39])
    ax_neighbor_waveforms=placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0.0,0.39])#[ax_neighbor_waveform_1,ax_neighbor_waveform_2,ax_neighbor_waveform_3]
    ax_CCGs_1 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.53,0.68],yspan=[0.36,0.48])
    ax_CCGs_2 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.7,.85],yspan=[0.36,0.48])
    ax_CCGs_3 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.86,1.0],yspan=[0.36,0.48])
    ax_CCGs = [ax_CCGs_1,ax_CCGs_2,ax_CCGs_3]
    
    #position plot
    ax_position.imshow(data[clusterID]['waveform_weights'][::4],extent=(site_positions[:,0][::4][0],site_positions[:,0][::4][0]+16,site_positions[:,1][::4][0],site_positions[:,1][::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.imshow(data[clusterID]['waveform_weights'][1::4],extent=(site_positions[:,0][1::4][0],site_positions[:,0][1::4][0]+16,site_positions[:,1][1::4][0],site_positions[:,1][1::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.imshow(data[clusterID]['waveform_weights'][2::4],extent=(site_positions[:,0][2::4][0],site_positions[:,0][2::4][0]+16,site_positions[:,1][2::4][0],site_positions[:,1][2::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.imshow(data[clusterID]['waveform_weights'][3::4],extent=(site_positions[:,0][3::4][0],site_positions[:,0][3::4][0]+16,site_positions[:,1][3::4][0],site_positions[:,1][3::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.set_aspect(0.1)
    ax_position.set_ylim(3840,0)
    ax_position.set_xlim(70,0)
    cleanAxes(ax_position)
    ax_position.set_title('neuron position')
    
    #time limits
    if time_limits == None:
        time_limits=[0,1e7]
    all_spikes = data[clusterID]['times']
    spike_times = all_spikes[np.where(all_spikes > time_limits[0])[0][0]:np.where(all_spikes < time_limits[1])[0][-1]]
    #print len(all_spikes)
    #print all_spikes[0]
    #print all_spikes[-1]
    ##print np.where(all_spikes > time_limits[0])[0][0]
    #print np.where(all_spikes < time_limits[1])[0][-1]
    these_spikes = np.where(cluster_IDs==cluster_ID)[0][np.where(all_spikes > time_limits[0])[0][0]:np.where(all_spikes < time_limits[1])[0][-1]]
    #spike_times = spike_times_data[these_spikes]

    #for PC and CCG display, find close by clusters calculate 
    #PC plot
    number_of_spikes_to_plot = 5000
    these_templates=spike_templates[np.where(cluster_IDs==cluster_ID)[0]]
    (included_templates,instances) = count_unique(these_templates)
    this_template = included_templates[int(np.where(instances==np.max(instances))[0])]
    ch1 = pc_ind_data[this_template][0]
    ch2 = pc_ind_data[this_template][1]
    ax_PCs.plot(pc_data[these_spikes][:number_of_spikes_to_plot,0,0],
             pc_data[these_spikes][:number_of_spikes_to_plot,0,1],'bo',ms=1.5,markeredgewidth=0)
    
    nearby_trio = [0,0,0];
    nearby_euclids = [10000,10000,10000];
    nearby_times=[]
    for other_cluster in data.keys():
        if other_cluster != clusterID:
            if (np.abs(data[clusterID]['ypos']-data[other_cluster]['ypos']) + np.abs(data[clusterID]['xpos']-data[other_cluster]['xpos'])) < nearby_euclids[-1]:
                rank = np.where((np.abs(data[clusterID]['ypos']-data[other_cluster]['ypos']) + np.abs(data[clusterID]['xpos']-data[other_cluster]['xpos'])) < nearby_euclids)[0][0]
                nearby_euclids[rank] = (np.abs(data[clusterID]['ypos']-data[other_cluster]['ypos']) + np.abs(data[clusterID]['xpos']-data[other_cluster]['xpos']))
                nearby_trio[rank]=other_cluster
    print(nearby_trio)
    for ii,neighbor in enumerate(nearby_trio):
        all_spikes_neighbor = data[neighbor]['times']
        indices_neighbor= np.where(cluster_IDs==int(neighbor))[0][np.where(all_spikes_neighbor > time_limits[0])[0][0]:np.where(all_spikes_neighbor < time_limits[1])[0][-1]]
        neighbor_templates=spike_templates[indices_neighbor]
        neighbor_spike_times = all_spikes_neighbor[np.where(all_spikes_neighbor > time_limits[0])[0][0]:np.where(all_spikes_neighbor < time_limits[1])[0][-1]]
        nearby_times.append(neighbor_spike_times)
        (included_templates,instances) = count_unique(neighbor_templates)
        this_template = included_templates[int(np.where(instances==np.max(instances))[0])]
        ch1_index = np.where(pc_ind_data[this_template] == ch1)[0]
        ch2_index = np.where(pc_ind_data[this_template] == ch2)[0]
        if ch2_index.size != 0 and ch1_index.size != 0:
            ax_PCs.plot(pc_data[indices_neighbor][:number_of_spikes_to_plot,0,np.where(pc_ind_data[this_template] == ch1)[0][0]],
                     pc_data[indices_neighbor][:number_of_spikes_to_plot,0,np.where(pc_ind_data[this_template] == ch2)[0][0]],
                     'o',color=neighbor_colors[ii],ms=1,markeredgewidth=0,alpha=0.8)
            
            all_diffs = []
            for spike_time in spike_times:
                try:
                    neighbor_start = np.where(neighbor_spike_times < spike_time - 0.5)[0][-1]
                except:
                    neighbor_start = 0
                try:
                    neighbor_end = np.where(neighbor_spike_times > spike_time + 0.5)[0][0]
                except:
                    neighbor_end = -1
                neighbor_chunk = neighbor_spike_times[neighbor_start:neighbor_end]
                #print '\r'+str(spike_time)+' '+str(neighbor_start)+' - '+str(neighbor_end),
                all_diffs.extend(neighbor_chunk - spike_time)
            all_diffs=np.array(all_diffs).flatten()
            hist,edges = np.histogram(all_diffs,bins=np.linspace(-0.2,0.2,400))
            ax_CCGs[ii].plot(edges[:-1],hist,drawstyle='steps',color=neighbor_colors[ii])
            ax_CCGs[ii].set_xlim(-.02,.02)
            ax_CCGs[ii].xaxis.set_ticks([-0.075,0.0,0.075])
            ax_CCGs[ii].axvline(0.0,ymax=0.1,ls='--',color='#ff8080')
        cleanAxes(ax_CCGs[ii],total=True)
    cleanAxes(ax_neighbor_waveforms,total=True)
    cleanAxes(ax_PCs)    
    ax_PCs.set_title('PC features')
    
    #waveform plot
    cleanAxes(ax_waveform,total=True)
    ax_waveform.set_title('waveform')
    channel_offset = 0 # should not be 0 if not starting from tip.
    peak_y_channel = np.where(data[clusterID]['waveform_weights'] == np.max(data[clusterID]['waveform_weights']))[0][0]
    times = np.random.choice(spike_times,100,replace=False)
    random_times = np.random.rand(100) * (np.max(spike_times)-np.min(spike_times)) + np.min(spike_times)
    xoffs=[0,100,50,150,0,100,50,150,0,100,50,150,0,100,50,150,0,100][::-1]
    yoff=100
    signal = 0;
    for ii,channel in enumerate(np.linspace(peak_y_channel-8,peak_y_channel+10,18)):
        if channel > 0:
            ws = load_waveforms(datapath,channel,times,channels=channels)
            ws_bkd = load_waveforms(datapath,channel,random_times,channels=channels)
            x_range = np.linspace(xoffs[ii],xoffs[ii]+60,60)
            for i in range(np.shape(ws)[0]):
                if np.shape(x_range)[0] == np.shape(ws[i][:]-yoff*(ii/2))[0]:
                    ax_waveform.plot(x_range,ws[i][:]-yoff*(ii/2),alpha=0.05,color='#0066ff')
                    ax_waveform.plot(x_range,ws_bkd[i][:]-yoff*(ii/2),alpha=0.05,color='#c8c8c8')
            if np.shape(x_range)[0] == np.shape(np.mean(ws,axis=0)-yoff*(ii/2))[0]:
                if np.max(np.abs(np.mean(ws,axis=0)-yoff*(ii/2))) > signal:
                    signal = np.max(np.abs(np.mean(ws,axis=0)-yoff*(ii/2)))
                    noise = np.mean(ws_bkd) + np.std(ws_bkd)* 4.
                ax_waveform.plot(x_range,np.mean(ws,axis=0)-yoff*(ii/2),color='#0066ff')
            if ii > 3 and ii < 14:
                for nn,axis in enumerate(nearby_trio):
                        if np.shape(x_range)[0] == np.shape(np.mean(ws,axis=0)-yoff*(ii/2))[0]:
                            neighbor_ws = load_waveforms(datapath,channel,np.random.choice(nearby_times[nn],100,replace=False),channels=channels)
                            ax_neighbor_waveforms.plot(x_range,np.mean(ws,axis=0)-(yoff/3.)*(ii/2),color='#0066ff')
                            ax_neighbor_waveforms.plot(x_range,np.mean(neighbor_ws,axis=0)-(yoff/3.)*(ii/2),color=neighbor_colors[nn],alpha=0.8)

    #time plot
    hist,edges = np.histogram(spike_times,bins=int(np.ceil(spike_times[-1] / timeplot_binsize)))
    ax_time.plot(edges[1:],hist/float(timeplot_binsize),drawstyle='steps')  
    ax_time.set_xlabel('time (sec)')
    ax_time.set_ylabel('firing rate (Hz)')
    ax_time.set_title('firing rate over time')
    max_spikes_to_plot = 2000
    if len(spike_times) > max_spikes_to_plot:
        times = np.random.choice(spike_times,max_spikes_to_plot,replace=False)
    else:
        times = spike_times
    amps = get_spike_amplitudes(datapath,peak_y_channel,times,channels=channels)
    ax_time_r = ax_time.twinx()
    ax_time_r.plot(times,amps,'o',markersize=2,alpha=0.3) 
    ax_time_r.set_ylabel('amplitude')
    ax_time_r.set_ylim(0,np.max(amps))
    cleanAxes(ax_time,leftLabels=True,bottomLabels=True)
    cleanAxes(ax_time_r,rightLabels=True,bottomLabels=True)
    
    #for ACG display, calculate ISIs.
    isis = np.diff(spike_times)
    hist,edges = np.histogram(np.concatenate((isis,isis*-1)),bins=np.linspace(-0.5,0.5,100))
    ax_ACG[0].plot(edges[:-1],hist,drawstyle='steps')
    ax_ACG[0].set_xlim(-.5,.5)
    ax_ACG[0].xaxis.set_ticks([-0.25,0.0,0.25])
    ax_ACG[0].set_ylabel('spike count')
    hist,edges = np.histogram(np.concatenate((isis,isis*-1)),bins=np.linspace(-0.02,0.02,160))
    ax_ACG[1].plot(edges[:-1],hist,drawstyle='steps')
    ax_ACG[1].set_xlim(-.02,.02)
    ax_ACG[1].axvline(0.0015,ls='--',color='#ff8080');ax_ACG[1].axvline(-0.0015,ls='--',color='#ff8080')
    ax_ACG[1].xaxis.set_ticks([-0.02,-0.01,0.0,0.01,0.02])
    for axis in ax_ACG:
        axis.set_xlabel('time (sec)')
    
    
    #text info plot
    cleanAxes(ax_text,total=True)
    ax_text.text(0, 1, 'cluster: '+clusterID, fontsize=12,weight='bold')
    #"%.2f" % signal
    #ax_text.text(10, 30, 'amp.: '+"%.2f" % 1.0+'uV', fontsize=10)
    #ax_text.text(10, 60, 'SNR: '+"%.2f" % float(df[df.clusterID==int(clusterID)].sn_max), fontsize=10)
    #ax_text.text(50, 0, 'isolation distance: '+"%.2f" % float(df[df.clusterID==int(clusterID)].isolation_distance), fontsize=10)
    #ax_text.text(50, 30, 'purity [mahalanobis]: '+"%.2f" % float(1-float(df[df.clusterID==int(clusterID)].mahalanobis_contamination)), fontsize=10)
    #ax_text.text(50, 60, 'ISI violation rate: '+"%.2f" % float(1-float(df[df.clusterID==int(clusterID)].isi_purity)), fontsize=10)
    ax_text.set_ylim(100,0)
    ax_text.set_xlim(0,100)
    
    return plt.gcf()
##===============================================================================
##===============================================================================

# %%



































##===============================================================================
##===============================================================================
##===============================================================================
##===============================================================================
##===============================================================================
##OLD PLOTTING FUNCTIONS
##===============================================================================
##===============================================================================
##===============================================================================
##===============================================================================
def plot_quality(quality,isiV,labels):
    (good,mua,unsorted) = labels
    f,ax = plt.subplots(3,1,figsize=(4,6));
    ax[0].plot(quality[1][np.where(np.in1d(quality[0],good))[0]],quality[2][np.where(np.in1d(quality[0],good))[0]],'o',label='good',mfc='g',mec='g');#plt.ylim(0,1)
    ax[0].plot(quality[1][np.where(np.in1d(quality[0],mua))[0]],quality[2][np.where(np.in1d(quality[0],mua))[0]],'o',label='mua',mfc='r',mec='r');#plt.ylim(0,1)
    ax[0].plot(quality[1][np.where(np.in1d(quality[0],unsorted))[0]],quality[2][np.where(np.in1d(quality[0],unsorted))[0]],'k.',label='unsorted',alpha=0.3);#plt.ylim(0,1)
    ax[0].set_xlabel('iso distance');ax[0].set_ylabel('contamination from mahal.')
    ax[0].set_ylim(ymin=0);ax[0].set_xlim(xmin=1.)
    ax[0].set_xscale('log');
    # if np.max(quality[1]) > 20:
    #     ax[0].set_xlim(1,20)
    legend = ax[0].legend(loc='upper right', shadow=False, fontsize=10,numpoints=1)

    ax[1].plot(quality[1][np.where(np.in1d(quality[0],good))[0]],isiV[1][np.where(np.in1d(isiV[0],good))[0]],'o',label='good',mfc='g',mec='g');plt.ylim(0,1)
    ax[1].plot(quality[1][np.where(np.in1d(quality[0],mua))[0]],isiV[1][np.where(np.in1d(isiV[0],mua))[0]],'o',label='mua',mfc='r',mec='r');plt.ylim(0,1)
    ax[1].plot(quality[1][np.where(np.in1d(quality[0],unsorted))[0]],isiV[1][np.where(np.in1d(isiV[0],unsorted))[0]],'k.',label='unsorted',alpha=0.3);plt.ylim(0,1)
    ax[1].set_xlabel('iso distance');ax[1].set_ylabel('isi contamination')
    ax[1].set_ylim(ymin=0);ax[1].set_xlim(xmin=1.)
    ax[1].set_xscale('log');
    # if np.max(quality[1]) > 20:
    #     ax[1].set_xlim(1,20)
    legend = ax[1].legend(loc='upper right', shadow=False, fontsize=10,numpoints=1)

    ax[2].plot(quality[2][np.where(np.in1d(quality[0],good))[0]],isiV[1][np.where(np.in1d(isiV[0],good))[0]],'o',label='good',mfc='g',mec='g');plt.ylim(0,1)
    ax[2].plot(quality[2][np.where(np.in1d(quality[0],mua))[0]],isiV[1][np.where(np.in1d(isiV[0],mua))[0]],'o',label='mua',mfc='r',mec='r');plt.ylim(0,1)
    ax[2].plot(quality[2][np.where(np.in1d(quality[0],unsorted))[0]],isiV[1][np.where(np.in1d(isiV[0],unsorted))[0]],'k.',label='unsorted',alpha=0.3);plt.ylim(0,1)
    ax[2].set_ylabel('isi contamination');ax[2].set_xlabel('contamination from mahal.')
    ax[2].set_ylim(ymin=0.01);ax[2].set_xlim(xmin=0.01)
    ax[2].set_xscale('log');ax[2].set_yscale('log')
    #legend = ax[2].legend(loc='upper left', shadow=False, fontsize=10,numpoints=1)
    
    plt.tight_layout()

def neuron_fig2(clusterID,df,sortpath,filename='experiment1_102-0_0.dat',time_limits=None,timeplot_binsize=60.,neighbor_colors=["#67572e","#50a874","#ff4d4d"]):
    channels = get_channel_count(sortpath)
    site_positions=option234_positions[np.linspace(0,channels-1,channels).astype(int)]
    data = load_phy_template(sortpath,site_positions)
    
    cluster_ID = int(clusterID)
    cluster_IDs = np.load(os.path.join(sortpath,'spike_clusters.npy'))
    pc_data = np.load(os.path.join(sortpath,'pc_features.npy'))
    pc_ind_data = np.load(os.path.join(sortpath,'pc_feature_ind.npy'))
    params_path = os.path.join(sortpath,'params.py')
    params = read_kilosort_params(params_path)
    spike_times_data = np.load(os.path.join(sortpath,'spike_times.npy'))/ float(params['sample_rate'])
    spike_templates = np.load(os.path.join(sortpath,'spike_templates.npy'))
    
    datapath=os.path.join(sortpath,'experiment1_102-0_0.dat')
    df_line = df[df.clusterID==int(clusterID)]
    
    fig = plt.figure(figsize=(11,8.5))
    ax_text = placeAxesOnGrid(fig,xspan=[0.0,0.4],yspan=[0,0.1])
    ax_position = placeAxesOnGrid(fig,xspan=[0,0.1],yspan=[0.12,1.0])
    ax_waveform = placeAxesOnGrid(fig,xspan=[0.2,0.45],yspan=[0.12,0.65])
    ax_time = placeAxesOnGrid(fig,xspan=[0.2,1.0],yspan=[0.82,1.0])
    ax_PCs = placeAxesOnGrid(fig,xspan=[0.5,0.8],yspan=[0,0.35])
    ax_ACG = placeAxesOnGrid(fig,dim=[1,2],xspan=[0.55,1.0],yspan=[0.5,0.7])
    #ax_neighbor_waveform_1 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0,0.13])
    #ax_neighbor_waveform_2 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0.13,0.26])
    #ax_neighbor_waveform_3 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0.26,0.39])
    ax_neighbor_waveforms=placeAxesOnGrid(fig,dim=[1,1],xspan=[0.82,1.0],yspan=[0.0,0.39])#[ax_neighbor_waveform_1,ax_neighbor_waveform_2,ax_neighbor_waveform_3]
    ax_CCGs_1 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.53,0.68],yspan=[0.36,0.48])
    ax_CCGs_2 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.7,.85],yspan=[0.36,0.48])
    ax_CCGs_3 = placeAxesOnGrid(fig,dim=[1,1],xspan=[0.86,1.0],yspan=[0.36,0.48])
    ax_CCGs = [ax_CCGs_1,ax_CCGs_2,ax_CCGs_3]
    
    #position plot
    ax_position.imshow(data[clusterID]['waveform_weights'][::4],extent=(site_positions[:,0][::4][0],site_positions[:,0][::4][0]+16,site_positions[:,1][::4][0],site_positions[:,1][::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.imshow(data[clusterID]['waveform_weights'][1::4],extent=(site_positions[:,0][1::4][0],site_positions[:,0][1::4][0]+16,site_positions[:,1][1::4][0],site_positions[:,1][1::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.imshow(data[clusterID]['waveform_weights'][2::4],extent=(site_positions[:,0][2::4][0],site_positions[:,0][2::4][0]+16,site_positions[:,1][2::4][0],site_positions[:,1][2::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.imshow(data[clusterID]['waveform_weights'][3::4],extent=(site_positions[:,0][3::4][0],site_positions[:,0][3::4][0]+16,site_positions[:,1][3::4][0],site_positions[:,1][3::4][-1]),cmap=plt.cm.gray_r,clim=(0,0.5),interpolation='none')
    ax_position.set_aspect(0.1)
    ax_position.set_ylim(3840,0)
    ax_position.set_xlim(70,0)
    cleanAxes(ax_position)
    ax_position.set_title('neuron position')
    
    #time limits
    if time_limits == None:
        time_limits=[0,1e7]
    all_spikes = data[clusterID]['times']
    spike_times = all_spikes[np.where(all_spikes > time_limits[0])[0][0]:np.where(all_spikes < time_limits[1])[0][-1]]
    #print len(all_spikes)
    #print all_spikes[0]
    #print all_spikes[-1]
    ##print np.where(all_spikes > time_limits[0])[0][0]
    #print np.where(all_spikes < time_limits[1])[0][-1]
    these_spikes = np.where(cluster_IDs==cluster_ID)[0][np.where(all_spikes > time_limits[0])[0][0]:np.where(all_spikes < time_limits[1])[0][-1]]
    #spike_times = spike_times_data[these_spikes]

    #for PC and CCG display, find close by clusters calculate 
    #PC plot
    number_of_spikes_to_plot = 2000
    these_templates=spike_templates[np.where(cluster_IDs==cluster_ID)[0]]
    (included_templates,instances) = count_unique(these_templates)
    this_template = included_templates[int(np.where(instances==np.max(instances))[0])]
    ch1 = pc_ind_data[this_template][0]
    ch2 = pc_ind_data[this_template][1]
    ax_PCs.plot(pc_data[these_spikes][:number_of_spikes_to_plot,0,0],
             pc_data[these_spikes][:number_of_spikes_to_plot,0,1],'bo',ms=1.5,markeredgewidth=0)
    
    nearby_trio = [0,0,0];
    nearby_euclids = [10000,10000,10000];
    nearby_times=[]
    for other_cluster in data.keys():
        if other_cluster != clusterID:
            if (np.abs(data[clusterID]['ypos']-data[other_cluster]['ypos']) + np.abs(data[clusterID]['xpos']-data[other_cluster]['xpos'])) < nearby_euclids[-1]:
                rank = np.where((np.abs(data[clusterID]['ypos']-data[other_cluster]['ypos']) + np.abs(data[clusterID]['xpos']-data[other_cluster]['xpos'])) < nearby_euclids)[0][0]
                nearby_euclids[rank] = (np.abs(data[clusterID]['ypos']-data[other_cluster]['ypos']) + np.abs(data[clusterID]['xpos']-data[other_cluster]['xpos']))
                nearby_trio[rank]=other_cluster
    print(nearby_trio)
    for ii,neighbor in enumerate(nearby_trio):
        all_spikes_neighbor = data[neighbor]['times']
        indices_neighbor= np.where(cluster_IDs==int(neighbor))[0][np.where(all_spikes_neighbor > time_limits[0])[0][0]:np.where(all_spikes_neighbor < time_limits[1])[0][-1]]
        neighbor_templates=spike_templates[indices_neighbor]
        neighbor_spike_times = all_spikes_neighbor[np.where(all_spikes_neighbor > time_limits[0])[0][0]:np.where(all_spikes_neighbor < time_limits[1])[0][-1]]
        nearby_times.append(neighbor_spike_times)
        (included_templates,instances) = count_unique(neighbor_templates)
        this_template = included_templates[int(np.where(instances==np.max(instances))[0])]
        ch1_index = np.where(pc_ind_data[this_template] == ch1)[0]
        ch2_index = np.where(pc_ind_data[this_template] == ch2)[0]
        if ch2_index.size != 0 and ch1_index.size != 0:
            ax_PCs.plot(pc_data[indices_neighbor][:number_of_spikes_to_plot,0,np.where(pc_ind_data[this_template] == ch1)[0][0]],
                     pc_data[indices_neighbor][:number_of_spikes_to_plot,0,np.where(pc_ind_data[this_template] == ch2)[0][0]],
                     'o',color=neighbor_colors[ii],ms=1,markeredgewidth=0,alpha=0.8)
            
            all_diffs = []
            for spike_time in spike_times:
                try:
                    neighbor_start = np.where(neighbor_spike_times < spike_time - 0.5)[0][-1]
                except:
                    neighbor_start = 0
                try:
                    neighbor_end = np.where(neighbor_spike_times > spike_time + 0.5)[0][0]
                except:
                    neighbor_end = -1
                neighbor_chunk = neighbor_spike_times[neighbor_start:neighbor_end]
                #print '\r'+str(spike_time)+' '+str(neighbor_start)+' - '+str(neighbor_end),
                all_diffs.extend(neighbor_chunk - spike_time)
            all_diffs=np.array(all_diffs).flatten()
            hist,edges = np.histogram(all_diffs,bins=np.linspace(-0.2,0.2,400))
            ax_CCGs[ii].plot(edges[:-1],hist,drawstyle='steps',color=neighbor_colors[ii])
            ax_CCGs[ii].set_xlim(-.2,.2)
            ax_CCGs[ii].xaxis.set_ticks([-0.075,0.0,0.075])
            ax_CCGs[ii].axvline(0.0,ls='--',color='#ff8080')
        cleanAxes(ax_CCGs[ii],total=True)
    cleanAxes(ax_neighbor_waveforms,total=True)
    cleanAxes(ax_PCs)    
    ax_PCs.set_title('PC features')
    
    #waveform plot
    cleanAxes(ax_waveform,total=True)
    ax_waveform.set_title('waveform')
    channel_offset = 0 # should not be 0 if not starting from tip.
    peak_y_channel = np.where(data[clusterID]['waveform_weights'] == np.max(data[clusterID]['waveform_weights']))[0][0]
    times = np.random.choice(spike_times,100,replace=False)
    random_times = np.random.rand(100) * (np.max(spike_times)-np.min(spike_times)) + np.min(spike_times)
    xoffs=[0,100,50,150,0,100,50,150,0,100,50,150,0,100,50,150,0,100][::-1]
    yoff=1500
    signal = 0;
    for ii,channel in enumerate(np.linspace(peak_y_channel-8,peak_y_channel+10,18)):
        ws = load_waveforms(datapath,channel,times,channels=channels)
        ws_bkd = load_waveforms(datapath,channel,random_times,channels=channels)
        x_range = np.linspace(xoffs[ii],xoffs[ii]+60,60)
        for i in range(np.shape(ws)[0]):
            if np.shape(x_range)[0] == np.shape(ws[i][:]-yoff*(ii/2))[0]:
                ax_waveform.plot(x_range,ws[i][:]-yoff*(ii/2),alpha=0.05,color='#0066ff')
                ax_waveform.plot(x_range,ws_bkd[i][:]-yoff*(ii/2),alpha=0.05,color='#c8c8c8')
        if np.shape(x_range)[0] == np.shape(np.mean(ws,axis=0)-yoff*(ii/2))[0]:
            if np.max(np.abs(np.mean(ws,axis=0)-yoff*(ii/2))) > signal:
                signal = np.max(np.abs(np.mean(ws,axis=0)-yoff*(ii/2)))
                noise = np.mean(ws_bkd) + np.std(ws_bkd)* 4.
            ax_waveform.plot(x_range,np.mean(ws,axis=0)-yoff*(ii/2),color='#0066ff')
        if ii > 3 and ii < 14:
            for nn,axis in enumerate(nearby_trio):
                    if np.shape(x_range)[0] == np.shape(np.mean(ws,axis=0)-yoff*(ii/2))[0]:
                        neighbor_ws = load_waveforms(datapath,channel,np.random.choice(nearby_times[nn],100,replace=False),channels=channels)
                        ax_neighbor_waveforms.plot(x_range,np.mean(ws,axis=0)-yoff*(ii/2),color='#0066ff')
                        ax_neighbor_waveforms.plot(x_range,np.mean(neighbor_ws,axis=0)-yoff*(ii/2),color=neighbor_colors[nn],alpha=0.8)



    #time plot
    hist,edges = np.histogram(spike_times,bins=np.ceil(spike_times[-1] / timeplot_binsize))
    ax_time.plot(edges[1:],hist/float(timeplot_binsize),drawstyle='steps')  
    ax_time.set_xlabel('time (sec)')
    ax_time.set_ylabel('firing rate (Hz)')
    ax_time.set_title('firing rate over time')
    max_spikes_to_plot = 2000
    if len(spike_times) > max_spikes_to_plot:
        times = np.random.choice(spike_times,max_spikes_to_plot,replace=False)
    else:
        times = spike_times
    amps = get_spike_amplitudes(datapath,peak_y_channel,times)
    ax_time_r = ax_time.twinx()
    ax_time_r.plot(times,amps,'o',markersize=2,alpha=0.1) 
    ax_time_r.set_ylabel('ampltiude')
    cleanAxes(ax_time,leftLabels=True,bottomLabels=True)
    cleanAxes(ax_time_r,rightLabels=True,bottomLabels=True)
    
    #for ACG display, calculate ISIs.
    isis = np.diff(spike_times)
    hist,edges = np.histogram(np.concatenate((isis,isis*-1)),bins=np.linspace(-0.5,0.5,100))
    ax_ACG[0].plot(edges[:-1],hist,drawstyle='steps')
    ax_ACG[0].set_xlim(-.5,.5)
    ax_ACG[0].xaxis.set_ticks([-0.25,0.0,0.25])
    ax_ACG[0].set_ylabel('spike count')
    hist,edges = np.histogram(np.concatenate((isis,isis*-1)),bins=np.linspace(-0.02,0.02,160))
    ax_ACG[1].plot(edges[:-1],hist,drawstyle='steps')
    ax_ACG[1].set_xlim(-.02,.02)
    ax_ACG[1].axvline(0.0015,ls='--',color='#ff8080');ax_ACG[1].axvline(-0.0015,ls='--',color='#ff8080')
    ax_ACG[1].xaxis.set_ticks([-0.02,-0.01,0.0,0.01,0.02])
    for axis in ax_ACG:
        axis.set_xlabel('time (sec)')
    
    
    #text info plot
    cleanAxes(ax_text,total=True)
    ax_text.text(0, 1, 'cluster: '+clusterID, fontsize=12,weight='bold')
    "%.2f" % signal
    ax_text.text(10, 30, 'amp.: '+"%.2f" % signal+'uV', fontsize=10)
    ax_text.text(10, 60, 'SNR: '+"%.2f" % (signal/noise), fontsize=10)
    #ax_text.text(50, 0, 'isolation distance: '+"%.2f" % quality[1][np.where(quality[0]==int(clusterID))[0][0]], fontsize=10)
    #ax_text.text(50, 30, 'purity [mahalanobis]: '+"%.2f" % quality[2][np.where(quality[0]==int(clusterID))[0][0]], fontsize=10)
    #ax_text.text(50, 60, 'ISI violation rate: '+"%.2f" % isiV[1][np.where(isiV[0]==int(clusterID))[0][0]]+'%', fontsize=10)
    ax_text.set_ylim(100,0)
    ax_text.set_xlim(0,100)
    
    return plt.gcf()
##===============================================================================
##===============================================================================
##===============================================================================
##===============================================================================
##===============================================================================