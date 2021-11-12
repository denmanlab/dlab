import numpy as np
import pandas as pd
import warnings
import glob, os, h5py, csv
from dlab.generalephys import option234_positions
from dlab.sglx_analysis import readAPMeta

try:
    from nwb.nwb import NWB
    from nwb.nwbts import TimeSeries
except:
    print('no Allen Institute NWB API. get this from http://stash.corp.alleninstitute.org/projects/INF/repos/ainwb/browse')
try:
    from djd.OpenEphys import loadContinuous, loadFolder, load_kwik_klusters
except:
    try:
        from OpenEphys import loadContinuous, loadFolder, load_kwik_klusters
    except:
        print('no OpenEphys.py get this from https://github.com/open-ephys/analysis-tools')

warnings.simplefilter(action='ignore', category=FutureWarning)

# def remap_data(data,electrodemap):
#     if type(data) - dict:
#         out = np.zeros(np.shape(data[data.keys()[10]])[0],len(electrodemap))
#         for ch in electrodemap:
#             out[:,ch] = data[np.sort(data.keys())[ch]]
#         return out

# class ExtracellularEphysFile(NWB):
#     def __init__(self,**kwargs):
#         modify=False
#         if 'meta' in kwargs.keys():
#             if 'modify' in kwargs.keys():
#                 modify=True
#             NWB.__init__(self,modify=modify,**kwargs['meta'])
#             self.meta = kwargs['meta']
#         else:
#             print('please provide information about the ephys data. use meta = {information about data, including at least "sort_method" and "sampling_rate"}')
#             return None
        
#     # THIS DOESN'T REALLY WORK for most kinds of data.
#     def add_raw_ephys(self,path,electrodemap=None,group=None,samplingrate=None,name='raw extracellualr ephys data'):
#         if samplingrate == None:
#             samplingrate = self.meta['sampling_rate']
#         if '.' in os.path.basename(path):
#             if os.path.basename(path).split('.')[-1] == 'dat':
#                 pass
            
#             if os.path.basename(path).split('.')[-1] == 'kwd':
#                 slf = self.file_pointer
#                 kwikfile = h5py.File(path,'r')
#                 if group == None:
#                     kwikfile.copy(kwikfile['recordings']['0'],slf['acquisition/timeseries'])
#                     slf.create_dataset('acquisition/timeseries/samplingrate',data=samplingrate)
#                 else:
#                     slf.create_group('acquisition/timeseries/'+group)
#                     kwikfile.copy(kwikfile['recordings']['0'],slf['acquisition/timeseries/'+group])
#                     slf.create_dataset('acquisition/timeseries/'+group+'/samplingrate',data=samplingrate)
                
#         else:     
#             data = loadFolder(path)
#             if electrodemap is not None:
#                 data = remap_data(data,electrodemap)
                
#             acquisition = self.create_timeseries('ElectricalSeries',name,'acquisition')
#             acquisition.set_data(data)
#             acquistion.close
    
#     # function for adding the clustering of some ephys data to an NWB file.
#     # takes a directory as input
#     # uses the 'sort_method' [defaults to the 'sort_method' in the metadata if no method provided.]
#     # options for 'sort method' are:
#     #       'phy':...placeholder, doesn't do anything right now...
#     #       'phy-template': expects cluster_groups.csv, spike_clusters.npy, and spike_times.npy in the directory
#     #       'kilosort': expects spike_templates.npy, and spike_times.npy in the directory. use only for the direct outputs of kilosort, that haven't been modified with phy-template
#     #       'spyking-circus': expects a .kwik and corresponding .kwx in the directory
#     #       'clu': expects spike time information (a dictionary with spiketimes for each unit) to be input with add_clustering(times=spiketime_information)
#     # places the results in the "processing" field of the NWB, under the name provided [default: 'extracellular electrophysiology data']
#     def add_clustering(self,path,**kwargs):
#         if 'sort_method' in kwargs.keys():
#             method = kwargs['sort_method']
#         else:
#             method = self.meta['sort_method']
            
#         if method == 'phy':
#             pass
#         if 'spyking-circus' in method:
#            self.spikes_data = load_kwik_klusters(path,phy_v=2)
#            self.spiketime_information = {}
#            self.all_spiketimes = []
#            self.all_cluster_numbers = []
#            for unit_name in self.spikes_data:
#               if type(self.spikes_data[unit_name]) == dict:
#                   if 'type' in self.spikes_data[unit_name].keys():
#                       if self.spikes_data[unit_name]['type'] == 'unit':
#                          self.spiketime_information[unit_name] = {}
#                          self.spiketime_information[unit_name]['spike_times'] = np.array(self.spikes_data[unit_name]['times'])/self.meta['sampling_rate']
#                          self.all_spiketimes.extend(np.array(np.array(self.spikes_data[unit_name]['times'])/self.meta['sampling_rate']).tolist())
#                          self.all_cluster_numbers.extend(np.array(np.ones(len(self.spikes_data[unit_name]['times']))*int(unit_name)).tolist())
        
#         if method == 'clu':
#            if 'times' in kwargs.keys():
#                spiketime_information = kwargs['times']
#            else:
#                raise ValueError('times not specified. use times = spiketime_information as an input to add_unit_times().')
        
#         if 'phy-template' in method:
#             self.all_spiketimes = []
#             self.all_cluster_numbers = []
            
#             self.clusters_data = np.load(open(os.path.join(path,'spike_clusters.npy')))
#             self.spikes_data = np.load(open(os.path.join(path,'spike_times.npy')))
#             spike_templates = np.load(open(os.path.join(path,'spike_templates.npy')))
#             templates = np.load(open(os.path.join(path,'templates.npy')))
            
#             cluster_groups = []
#             [cluster_groups.append(row) for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))];
#             self.spiketime_information = {}
#             for i in np.arange(1,np.shape(cluster_groups)[0]):
#                 if cluster_groups[i][0].split('\t')[1] == 'good':           #if it is a 'good' cluster by manual sort
#                     unit = int(cluster_groups[i][0].split('\t')[0])
#                     self.spiketime_information[str(unit)] = {}
                    
#                     self.spiketime_information[str(unit)]['spike_times'] = self.spikes_data[np.where(self.clusters_data==unit)]/self.meta['sampling_rate']    
#                     self.spiketime_information[str(unit)]['spike_times']= self.spiketime_information[str(unit)]['spike_times'].flatten()
                    
#                     #get the mean template used for this unit
#                     all_templates = spike_templates[np.where(self.clusters_data==unit)].flatten()
#                     n_templates_to_subsample = 100
#                     random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
#                     mean_template = np.mean(random_subsample_of_templates,axis=0)
#                     self.spiketime_information[str(unit)]['template'] = mean_template
                    
#                     #take a weighted average of the channelmap, where the weights is the absolute value of the template for that channel
#                     #this gets us the x and y positions of the unit on the probe.
#                     if 'channelmap' in kwargs.keys():
#                         channelmap = kwargs['channelmap']
#                         weights = np.zeros(channelmap.shape)
#                         for channel in range(channelmap.shape[0]):
#                             weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
#                         weights = weights/np.max(weights)
#                         (xpos,ypos)=np.average(channelmap,axis=0,weights=weights)
#                     else:
#                         (xpos,ypos)=(np.nan,np.nan)
#                     self.spiketime_information[str(unit)]['xpos'] = xpos + 6
#                     self.spiketime_information[str(unit)]['ypos'] = ypos - channelmap[-1][1]
                    
#                     self.all_spiketimes.extend(self.spiketime_information[str(unit)]['spike_times'])
#                     self.all_cluster_numbers.extend(np.array(np.ones(len(self.spiketime_information[str(unit)]['spike_times']))*unit).tolist())



			
#         if 'kilosort' == method:
#             self.all_spiketimes = []
#             self.all_cluster_numbers = []
#             self.clusters = np.load(open(os.path.join(path,'spike_clusters.npy')))
#             self.clusters_data = np.load(open(os.path.join(path,'spike_templates.npy')))
#             self.spikes_data = np.load(open(os.path.join(path,'spike_times.npy')))
#             self.templates =  np.load(open(os.path.join(path,'templates.npy')))
#             cluster_groups = []
#             if os.path.isfile(os.path.join(path,'cluster_groups.csv')):
#                 cluster_groups = [row for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))][1:];
#             else:
#                 [cluster_groups.append(str(row)+'\tgood') for row in np.unique(self.clusters_data)] # fake that all the clusters are good.
#             self.spiketime_information = {}
#             for i in np.arange(1,np.shape(cluster_groups)[0]):
#                 if cluster_groups[i][0].split('\t')[1] == 'good':           #if it is a 'good' cluster by manual sort
#                     unit = int(cluster_groups[i][0].split('\t')[0])
#                     self.spiketime_information[str(unit)] = {}
#                     self.spiketime_information[str(unit)]['spike_times'] = self.spikes_data[np.where(self.clusters==unit)]/self.meta['sampling_rate']    
#                     self.spiketime_information[str(unit)]['spike_times']= self.spiketime_information[str(unit)]['spike_times'].flatten()
#                     self.all_spiketimes.extend(self.spiketime_information[str(unit)]['spike_times'])
#                     self.all_cluster_numbers.extend(np.array(np.ones(len(self.spiketime_information[str(unit)]['spike_times']))*unit).tolist())
                    
#                     #get the mean template used for this unit
#                     if 'site_positions' in kwargs.keys():
#                         site_positions = kwargs['site_positions']
#                         if 'offset' in kwargs.keys():
#                             offset = kwargs['offset']
#                         else:
#                             offset = 0
#                         all_templates = self.clusters_data[np.where(self.clusters==unit)].flatten()
#                         n_templates_to_subsample = 100
#                         random_subsample_of_templates = self.templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
#                         mean_template = np.mean(random_subsample_of_templates,axis=0)
                        
#                         #take a weighted average of the site_positions, where the weights is the absolute value of the template for that channel
#                         #this gets us the x and y positions of the unit on the probe.
#                         weights = np.zeros(site_positions.shape)
#                         for channel in range(site_positions.shape[0]):
#                             weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
#                         weights = weights/np.max(weights)
#                         low_values_indices = weights < 0.25  # Where values are low,
#                         weights[low_values_indices] = 0      # make the weight 0
#                         (xpos,ypos)=np.average(site_positions,axis=0,weights=weights)
#                     else:
#                         mean_template,xpos,ypos = (np.nan,np.nan,np.nan)
#                     self.spiketime_information[str(unit)]['template'] = mean_template
#                     self.spiketime_information[str(unit)]['xpos'] = xpos
#                     self.spiketime_information[str(unit)]['ypos'] = ypos - offset
                    
                    
#         ##################################################################################################################
#         if 'name' in kwargs.keys():
#             name = kwargs['name']
#         else:
#             name = 'extracellular electrophysiology data'
#         self.mod = self.create_module(name)
        
#         #add clustering
#         clustering = self.mod.create_interface("Clustering")
#         clustering.set_clusters(self.all_spiketimes,self.all_cluster_numbers,np.zeros(len(self.all_cluster_numbers)))
#         clustering.finalize()
        
#         #add clustering_waveforms
#         #self.add_clustering_waveforms()
        
#         #add unit times
#         self.add_unit_times()


        
#         self.mod.finalize()
#         ##################################################################################################################
        
#     def add_unit_times(self,**kwargs):
#         unit_times = self.mod.create_interface("UnitTimes")
#         for unit_name in self.spiketime_information:
#             unit_times.add_unit(unit_name = unit_name, 
#                                 unit_times = np.sort(self.spiketime_information[unit_name]['spike_times']),
#                                 description = "All spiketimes are in SI units (seconds)",
#                                 source = "Data spike-sorted by: "+self.meta['user']+' using '+self.meta['sort_method'])
#             # also add waveform information, if it is avaialble, which for right now is just with phy-template
#             # now also with kilosort!
#             if 'phy-template' in  self.meta['sort_method'] or 'kilosort' in  self.meta['sort_method']:
#                 unit_times.append_unit_data(unit_name,'template',self.spiketime_information[unit_name]['template'])
#                 unit_times.append_unit_data(unit_name,'xpos',self.spiketime_information[unit_name]['xpos'])
#                 unit_times.append_unit_data(unit_name,'ypos',self.spiketime_information[unit_name]['ypos'])

#         unit_times.finalize()
        



        
#     def add_clustering_waveforms(self,**kwargs):
#         clustering_waveforms = self.mod.create_interface("ClusterWaveforms")
#         for unit_name in self.spiketime_information:
#             clustering_waveforms.add_waveform(cluster = unit_name,
#                                               waveform_mean = self.spikes_data[unit_name]['waveform'],
#                                               waveform_sd = self.spikes_data[unit_name]['waveform_sd'])
#         clustering_waveforms.finalize()
    
    
#     #add stimulus timing information to the ax        
#     def add_stimulus_information(self,timestamps,data,start_time=0.0,source='dome',name='visual stimulus - generic',**kwargs):
#         abstract = self.create_timeseries("AbstractFeatureSeries",name, "stimulus")
        
#         if 'features' in kwargs.keys():
#             if 'features_units' in kwargs.keys():
#                 abstract.set_features(kwargs['features'],kwargs['features_units'])
#             else:
#                 self.fatal_error("features_units is required when features is used")
#         abstract.set_data(data)
#         abstract.set_time(timestamps+start_time)
#         abstract.set_source(source)

#         abstract.finalize()

#     #add stimulus timing information to the ax        
#     def add_stimulus_template(self,data,times=np.array([0]),start_time=0.0,source='dome',name='visual stimulus - generic',**kwargs):
#         abstract = self.create_timeseries("AbstractFeatureSeries",name, "template")
        
#         if 'features' in kwargs.keys():
#             if 'features_units' in kwargs.keys():
#                 abstract.set_features(kwargs['features'],kwargs['features_units'])
#             else:
#                 self.fatal_error("features_units is required when features is used")
#         abstract.set_data(data)
#         abstract.set_source(source)
#         abstract.set_time(times+start_time)
#         abstract.finalize()
    
#     def add_eyedata(self,data,times,name='righteye',start_time=0.0):
#         self.eye_mod = self.create_module(name)
#         pupil_area = self.eye_mod.create_interface("PupilTracking")
#         pupil_area_ts = self.create_timeseries('TimeSeries','PupilArea')
#         pupil_area_ts.set_data(data['pupil_area'],conversion=np.array([1]).astype(np.float32)[0],resolution=np.array([1]).astype(np.float32)[0],unit='pixels')
#         pupil_area_ts.set_time(times+start_time)
#         pupil_area.add_timeseries(pupil_area_ts)
#         pupil_area_ts.finalize()
#         pupil_area.finalize()
#         eye_position = self.eye_mod.create_interface("EyeTracking")
#         eye_position_ts = self.create_timeseries('SpatialSeries','EyePositions')
#         eye_position_ts.set_data(data['pupil_positions'],conversion=np.array([1]).astype(np.float32)[0],resolution=np.array([1]).astype(np.float32)[0],unit='pixels')
#         eye_position_ts.set_value('reference_frame',['[0,0]'])
#         eye_position_ts.set_time(times+start_time)
#         eye_position.add_timeseries(eye_position_ts)
#         eye_position_ts.finalize()
#         eye_position.finalize()
        
#         self.eye_mod.finalize()
        
#     def add_to_general(self,data,name='general'):
#         self.set_metadata(name,data)
        
def load_phy_template(path,cluster_file='KS2',site_positions = option234_positions, **kwargs):
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
    cluster_id,KSlabel,KSamplitude,KScontamination = [],[],[],[]
    [KSlabel.append(row) for row in csv.reader(open(os.path.join(path,'cluster_KSLabel.tsv')))]
    [KSamplitude.append(row) for row in csv.reader(open(os.path.join(path,'cluster_Amplitude.tsv')))]
    [KScontamination.append(row) for row in csv.reader(open(os.path.join(path,'cluster_ContamPct.tsv')))]
    if os.path.isfile(os.path.join(path,'cluster_group.tsv')):
        # cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_group.tsv')))][1:]
        [cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_group.tsv')))]
    else:
        if os.path.isfile(os.path.join(path,'cluster_groups.csv')):
            # cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))][1:]
            [cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))]
        else: print('cant find cluster groups, either .tsv or .csv')
    if 'sampling_rate' in kwargs.keys():
        samplingrate = kwargs['sampling_rate']
    else:
        samplingrate =30000.
        # print('no sampling rate specified, using default of 30kHz')
        
    units = {}
    for i in np.arange(1,np.shape(cluster_id)[0]):
        unit = int(cluster_id[i][0].split('\t')[0])
        units[str(unit)] = {}
        
        #get the unit spike times
        units[str(unit)]['samples'] = spikes[np.where(clusters==unit)].flatten()
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
        # print(mean_template.T.shape)
        weights = np.zeros(site_positions.shape)
        for channel in range(mean_template.T.shape[0]):
            weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
        weights = weights/np.max(weights)
        low_values_indices = weights < 0.25  # Where values are low,
        weights[low_values_indices] = 0      # make the weight 0
        (xpos,ypos)=np.average(site_positions,axis=0,weights=weights)
        units[str(unit)]['waveform_weights'] = weights
        units[str(unit)]['xpos'] = xpos
        units[str(unit)]['ypos'] = ypos #- site_positions[-1][1]
        units[str(unit)]['label'] =  cluster_id[i][0].split('\t')[1]
        units[str(unit)]['KSlabel'] =  KSlabel[i][0].split('\t')[1]
        units[str(unit)]['KSamplitude'] =  KSamplitude[i][0].split('\t')[1]
        units[str(unit)]['KScontamination'] =  KScontamination[i][0].split('\t')[1]
    return units

def df_from_phy(path,site_positions = option234_positions,**kwargs):
    nwb_data = load_phy_template(path,site_positions)
    #structures is a dictionary that defines the bounds of the structure e.g.:{'v1':(0,850), 'hpc':(850,2000)}
    mouse = [];experiment=[];cell = [];ypos = [];xpos = [];waveform=[];template=[];structure=[];times=[]
    index = []; count = 1
    nwb_id = [];probe_id=[]
    depth=[];#print(list(nwb_data.keys()));print(list(nwb_data['processing'].keys()));
    if 'probe' in kwargs.keys():
        for probe in list(nwb_data['processing'].keys()):
            if 'UnitTimes' in list(nwb_data['processing'][probe].keys()): 
                for i,u in enumerate(list(nwb_data['processing'][probe]['UnitTimes'].keys())):
                    if u != 'unit_list':
                        nwb_id.append(nwbid)
                        probe_id.append(probe)
                        index.append(count);count+=1
                        mouse.append(str(np.array(nwb_data.get('identifier'))))
                        experiment.append(1)
                        cell.append(u)
                        times.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['times']));# print(list(nwb_data['processing'][probe]['UnitTimes'][u].keys()))
                        if 'ypos' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                            ypos.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['ypos']))
                            has_ypos = True
                        else:
                            ypos.append(None)
                            has_ypos = False				
                        if 'depth' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                            depth.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['depth']))
                        else:
                            if has_ypos:
                                depth.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['ypos']))
                            else:
                                depth.append(None)
                        if 'xpos' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                            xpos.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['xpos']))
                            has_xpos = True
                        else:
                            xpos.append(None)
                            has_xpos = False
                        template.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['template']))
                        waveform.append(get_peak_waveform_from_template(template[-1]))
                        if not structures == None:
                            structur = None
                            for struct, bounds in structures.iteritems():
                                if ypos[-1] > bounds[0] and ypos[-1]< bounds[1] :
                                    structur=struct
                        else:
                            structur = None
                        structure.append(structur)
    df = pd.DataFrame(index=index)
    df = df.fillna(np.nan)
    df['nwb_id'] = nwb_id
    df['mouse'] = mouse
    df['experiment'] = experiment
    df['probe'] = probe_id
    df['structure'] = structure
    df['cell'] = cell
    df['times'] = times
    df['ypos'] = ypos
    df['xpos'] = xpos
    df['depth'] = depth
    df['waveform'] = waveform
    df['template'] = template
    return df

def load_unit_data(recording_path, probe_depth = 3840, site_positions = option234_positions, 
                   probe_name=None, spikes_filename = 'spike_secs.npy', aligned=True, df=True, **kwargs):
# Inputs:
# Outputs:
    if probe_name == None: probe_name = recording_path
    #Get individual folders for each probe
    unit_times=[]
    if aligned == False:
#         if not sampling_rate:
#             imec_meta = readAPMeta(recording_path+'\\') #extract meta file
#             sampRate = float(imec_meta['imSampRate']) #get sampling rate (Hz)
#         else:
	    if 'sampling_rate' in kwargs.keys():
		    sampRate  = float(kwargs['sampling_rate'])
	    else:
		    sampRate=30000
	    spike_times = np.ndarray.flatten(np.load(os.path.join(recording_path, 'spike_times.npy')))/sampRate
    else:
        spike_times = np.ndarray.flatten(np.load(os.path.join(recording_path, spikes_filename)))

    cluster_info = pd.read_csv(os.path.join(recording_path, 'cluster_info.tsv'), '\t')
    if cluster_info.keys()[0]=='cluster_id':
        cluster_info = cluster_info.rename(columns={'cluster_id':'id'})
    spike_clusters = np.ndarray.flatten(np.load(os.path.join(recording_path, 'spike_clusters.npy')))
    spike_templates = np.load(open(os.path.join(recording_path,'spike_templates.npy'),'rb'))
    templates = np.load(open(os.path.join(recording_path,'templates.npy'),'rb'))
    amplitudes = np.load(open(os.path.join(recording_path,'amplitudes.npy'),'rb'))
    weights = np.zeros(site_positions.shape)

    #Generate Unit Times Table
    for index, unitID in enumerate(cluster_info['id'].values):
        #get mean template used for each unit
        all_templates = spike_templates[np.where(spike_clusters==unitID)].flatten()
        n_templates_to_subsample = 100
        random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
        mean_template = np.mean(random_subsample_of_templates,axis=0)

        #take a weighted average of the site_positions, where the weights is the absolute value of the template for that channel
        #this gets us the x and y positions of the unit on the probe.
        for channel in range(mean_template.T.shape[0]):
            weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
        weights = weights/np.max(weights)
        low_values_indices = weights < 0.25  # Where values are low,
        weights[low_values_indices] = 0      # make the weight 0
        (xpos,zpos)=np.average(site_positions,axis=0,weights=weights)

        unit_times.append({'probe':probe_name,
                           'unit_id': unitID,
                           'group': cluster_info.group[index],
#                                'depth':cluster_info.depth[index],
                           'depth': (zpos-3840)+probe_depth,
                           'xpos': xpos,
                           'zpos': zpos,
                           'no_spikes': cluster_info.n_spikes[index], 
                           'KSlabel': cluster_info['KSLabel'][index],
                           'KSamplitude':cluster_info.Amplitude[index],
                           'KScontamination': cluster_info.ContamPct[index],
                           'template': mean_template,
                           'waveform_weights': weights,
                           'amplitudes': amplitudes[:,0][spike_clusters==unitID],
                           'times': spike_times[spike_clusters == unitID],
                            })
    if df == True:        
        unit_data = pd.DataFrame(unit_times)
        #Remove clusters with no associated spike times left over from Phy
        for i,j in enumerate(unit_data.times):
            if len(unit_data.times[i])==0:
                unit_data.times[i]='empty'
        unit_times = unit_data[unit_data.times!='empty']
        return(unit_times)
    else:
        return(unit_times)

def multi_load_unit_data(recording_folder,probe_names=['A','B','C','D'],probe_depths=[3840,3840,3840,3840],spikes_filename = 'spike_secs.npy', aligned=True):
    folder_paths = glob.glob(os.path.join(recording_folder,'*imec*'))
    return pd.concat([load_unit_data(folder,probe_name=probe_names[i],probe_depth=probe_depths[i],spikes_filename = spikes_filename, aligned=True,df=True) for i,folder in enumerate(folder_paths)],ignore_index=True)
