import numpy as np
import glob, os, h5py, csv
try:
    from nwb.nwb import NWB
    from nwb.nwbts import TimeSeries
except:
    print 'no Allen Institute NWB API. get this from http://stash.corp.alleninstitute.org/projects/INF/repos/ainwb/browse'
try:
    from djd.OpenEphys import loadContinuous, loadFolder, load_kwik_klusters
except:
    try:
        from OpenEphys import loadContinuous, loadFolder, load_kwik_klusters
    except:
        print 'no OpenEphys.py get this from https://github.com/open-ephys/analysis-tools' 


def remap_data(data,electrodemap):
    if type(data) - dict:
        out = np.zeros(np.shape(data[data.keys()[10]])[0],len(electrodemap))
        for ch in electrodemap:
            out[:,ch] = data[np.sort(data.keys())[ch]]
        return out

class ExtracellularEphysFile(NWB):
    def __init__(self,**kwargs):
        modify=False
    	if 'meta' in kwargs.keys():
            if 'modify' in kwargs.keys():
                modify=True
            NWB.__init__(self,modify=modify,**kwargs['meta'])
            self.meta = kwargs['meta']
    	else:
            print 'please provide information about the ephys data. use meta = {information about data, including at least "sort_method" and "sampling_rate"}'
            return None
        
    # THIS DOESN'T REALLY WORK for most kinds of data.
    def add_raw_ephys(self,path,electrodemap=None,group=None,samplingrate=None,name='raw extracellualr ephys data'):
        if samplingrate == None:
            samplingrate = self.meta['sampling_rate']
        if '.' in os.path.basename(path):
            if os.path.basename(path).split('.')[-1] == 'dat':
                pass
            
            if os.path.basename(path).split('.')[-1] == 'kwd':
                slf = self.file_pointer
                kwikfile = h5py.File(path,'r')
                if group == None:
                    kwikfile.copy(kwikfile['recordings']['0'],slf['acquisition/timeseries'])
                    slf.create_dataset('acquisition/timeseries/samplingrate',data=samplingrate)
                else:
                    slf.create_group('acquisition/timeseries/'+group)
                    kwikfile.copy(kwikfile['recordings']['0'],slf['acquisition/timeseries/'+group])
                    slf.create_dataset('acquisition/timeseries/'+group+'/samplingrate',data=samplingrate)
                
        else:     
            data = loadFolder(path)
            if electrodemap is not None:
                data = remap_data(data,electrodemap)
                
            acquisition = self.create_timeseries('ElectricalSeries',name,'acquisition')
            acquisition.set_data(data)
            acquistion.close
    
    # function for adding the clustering of some ephys data to an NWB file.
    # takes a directory as input
    # uses the 'sort_method' [defaults to the 'sort_method' in the metadata if no method provided.]
    # options for 'sort method' are:
    #       'phy':...placeholder, doesn't do anything right now...
    #       'phy-template': expects cluster_groups.csv, spike_clusters.npy, and spike_times.npy in the directory
    #       'kilosort': expects spike_templates.npy, and spike_times.npy in the directory. use only for the direct outputs of kilosort, that haven't been modified with phy-template
    #       'spyking-circus': expects a .kwik and corresponding .kwx in the directory
    #       'clu': expects spike time information (a dictionary with spiketimes for each unit) to be input with add_clustering(times=spiketime_information)
    # places the results in the "processing" field of the NWB, under the name provided [default: 'extracellular electrophysiology data']
    def add_clustering(self,path,**kwargs):
        if 'sort_method' in kwargs.keys():
            method = kwargs['sort_method']
        else:
            method = self.meta['sort_method']
            
        if method == 'phy':
            pass
        if 'spyking-circus' in method:
           self.spikes_data = load_kwik_klusters(path,phy_v=2)
           self.spiketime_information = {}
           self.all_spiketimes = []
           self.all_cluster_numbers = []
           for unit_name in self.spikes_data:
              if type(self.spikes_data[unit_name]) == dict:
                  if 'type' in self.spikes_data[unit_name].keys():
                      if self.spikes_data[unit_name]['type'] == 'unit':
                         self.spiketime_information[unit_name] = {}
                         self.spiketime_information[unit_name]['spike_times'] = np.array(self.spikes_data[unit_name]['times'])/self.meta['sampling_rate']
                         self.all_spiketimes.extend(np.array(np.array(self.spikes_data[unit_name]['times'])/self.meta['sampling_rate']).tolist())
                         self.all_cluster_numbers.extend(np.array(np.ones(len(self.spikes_data[unit_name]['times']))*int(unit_name)).tolist())
        
        if method == 'clu':
           if 'times' in kwargs.keys():
               spiketime_information = kwargs['times']
           else:
               raise ValueError('times not specified. use times = spiketime_information as an input to add_unit_times().')
        
        if 'phy-template' in method:
            self.all_spiketimes = []
            self.all_cluster_numbers = []
            
            self.clusters_data = np.load(open(os.path.join(path,'spike_clusters.npy')))
            self.spikes_data = np.load(open(os.path.join(path,'spike_times.npy')))
            spike_templates = np.load(open(os.path.join(path,'spike_templates.npy')))
            templates = np.load(open(os.path.join(path,'templates.npy')))
            
            cluster_groups = []
            [cluster_groups.append(row) for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))];
            self.spiketime_information = {}
            for i in np.arange(1,np.shape(cluster_groups)[0]):
                if cluster_groups[i][0].split('\t')[1] == 'good':           #if it is a 'good' cluster by manual sort
                    unit = int(cluster_groups[i][0].split('\t')[0])
                    self.spiketime_information[str(unit)] = {}
                    
                    self.spiketime_information[str(unit)]['spike_times'] = self.spikes_data[np.where(self.clusters_data==unit)]/self.meta['sampling_rate']    
                    self.spiketime_information[str(unit)]['spike_times']= self.spiketime_information[str(unit)]['spike_times'].flatten()
                    
                    #get the mean template used for this unit
                    all_templates = spike_templates[np.where(self.clusters_data==unit)].flatten()
                    n_templates_to_subsample = 100
                    random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
                    mean_template = np.mean(random_subsample_of_templates,axis=0)
                    self.spiketime_information[str(unit)]['template'] = mean_template
                    
                    #take a weighted average of the channelmap, where the weights is the absolute value of the template for that channel
                    #this gets us the x and y positions of the unit on the probe.
                    if 'channelmap' in kwargs.keys():
                        channelmap = kwargs['channelmap']
                        weights = np.zeros(channelmap.shape)
                        for channel in range(channelmap.shape[0]):
                            weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
                        weights = weights/np.max(weights)
                        (xpos,ypos)=np.average(channelmap,axis=0,weights=weights)
                    else:
                        (xpos,ypos)=(np.nan,np.nan)
                    self.spiketime_information[str(unit)]['xpos'] = xpos + 6
                    self.spiketime_information[str(unit)]['ypos'] = ypos - channelmap[-1][1]
                    
                    self.all_spiketimes.extend(self.spiketime_information[str(unit)]['spike_times'])
                    self.all_cluster_numbers.extend(np.array(np.ones(len(self.spiketime_information[str(unit)]['spike_times']))*unit).tolist())



			
        if 'kilosort' == method:
            self.all_spiketimes = []
            self.all_cluster_numbers = []
            self.clusters = np.load(open(os.path.join(path,'spike_clusters.npy')))
            self.clusters_data = np.load(open(os.path.join(path,'spike_templates.npy')))
            self.spikes_data = np.load(open(os.path.join(path,'spike_times.npy')))
            self.templates =  np.load(open(os.path.join(path,'templates.npy')))
            cluster_groups = []
            if os.path.isfile(os.path.join(path,'cluster_groups.csv')):
                cluster_groups = [row for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))][1:];
            else:
                [cluster_groups.append(str(row)+'\tgood') for row in np.unique(self.clusters_data)] # fake that all the clusters are good.
            self.spiketime_information = {}
            for i in np.arange(1,np.shape(cluster_groups)[0]):
                if cluster_groups[i][0].split('\t')[1] == 'good':           #if it is a 'good' cluster by manual sort
                    unit = int(cluster_groups[i][0].split('\t')[0])
                    self.spiketime_information[str(unit)] = {}
                    self.spiketime_information[str(unit)]['spike_times'] = self.spikes_data[np.where(self.clusters==unit)]/self.meta['sampling_rate']    
                    self.spiketime_information[str(unit)]['spike_times']= self.spiketime_information[str(unit)]['spike_times'].flatten()
                    self.all_spiketimes.extend(self.spiketime_information[str(unit)]['spike_times'])
                    self.all_cluster_numbers.extend(np.array(np.ones(len(self.spiketime_information[str(unit)]['spike_times']))*unit).tolist())
                    
                    #get the mean template used for this unit
                    if 'site_positions' in kwargs.keys():
                        site_positions = kwargs['site_positions']
                        if 'offset' in kwargs.keys():
                            offset = kwargs['offset']
                        else:
                            offset = 0
                        all_templates = self.clusters_data[np.where(self.clusters==unit)].flatten()
                        n_templates_to_subsample = 100
                        random_subsample_of_templates = self.templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
                        mean_template = np.mean(random_subsample_of_templates,axis=0)
                        
                        #take a weighted average of the site_positions, where the weights is the absolute value of the template for that channel
                        #this gets us the x and y positions of the unit on the probe.
                        weights = np.zeros(site_positions.shape)
                        for channel in range(site_positions.shape[0]):
                            weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
                        weights = weights/np.max(weights)
                        low_values_indices = weights < 0.25  # Where values are low,
                        weights[low_values_indices] = 0      # make the weight 0
                        (xpos,ypos)=np.average(site_positions,axis=0,weights=weights)
                    else:
                        mean_template,xpos,ypos = (np.nan,np.nan,np.nan)
                    self.spiketime_information[str(unit)]['template'] = mean_template
                    self.spiketime_information[str(unit)]['xpos'] = xpos
                    self.spiketime_information[str(unit)]['ypos'] = ypos - offset
                    
                    
        ##################################################################################################################
        if 'name' in kwargs.keys():
            name = kwargs['name']
        else:
            name = 'extracellular electrophysiology data'
        self.mod = self.create_module(name)
        
        #add clustering
        clustering = self.mod.create_interface("Clustering")
        clustering.set_clusters(self.all_spiketimes,self.all_cluster_numbers,np.zeros(len(self.all_cluster_numbers)))
        clustering.finalize()
        
        #add clustering_waveforms
        #self.add_clustering_waveforms()
        
        #add unit times
        self.add_unit_times()


        
        self.mod.finalize()
        ##################################################################################################################
        
    def add_unit_times(self,**kwargs):
        unit_times = self.mod.create_interface("UnitTimes")
        for unit_name in self.spiketime_information:
            unit_times.add_unit(unit_name = unit_name, 
                                unit_times = np.sort(self.spiketime_information[unit_name]['spike_times']),
                                description = "All spiketimes are in SI units (seconds)",
                                source = "Data spike-sorted by: "+self.meta['user']+' using '+self.meta['sort_method'])
            # also add waveform information, if it is avaialble, which for right now is just with phy-template
            # now also with kilosort!
            if 'phy-template' in  self.meta['sort_method'] or 'kilosort' in  self.meta['sort_method']:
                unit_times.append_unit_data(unit_name,'template',self.spiketime_information[unit_name]['template'])
                unit_times.append_unit_data(unit_name,'xpos',self.spiketime_information[unit_name]['xpos'])
                unit_times.append_unit_data(unit_name,'ypos',self.spiketime_information[unit_name]['ypos'])

        unit_times.finalize()
        



        
    def add_clustering_waveforms(self,**kwargs):
        clustering_waveforms = self.mod.create_interface("ClusterWaveforms")
        for unit_name in self.spiketime_information:
            clustering_waveforms.add_waveform(cluster = unit_name,
                                              waveform_mean = self.spikes_data[unit_name]['waveform'],
                                              waveform_sd = self.spikes_data[unit_name]['waveform_sd'])
        clustering_waveforms.finalize()
    
    
    #add stimulus timing information to the ax        
    def add_stimulus_information(self,timestamps,data,start_time=0.0,source='dome',name='visual stimulus - generic',**kwargs):
        abstract = self.create_timeseries("AbstractFeatureSeries",name, "stimulus")
        
        if 'features' in kwargs.keys():
            if 'features_units' in kwargs.keys():
                abstract.set_features(kwargs['features'],kwargs['features_units'])
            else:
                self.fatal_error("features_units is required when features is used")
        abstract.set_data(data)
        abstract.set_time(timestamps+start_time)
        abstract.set_source(source)

        abstract.finalize()

    #add stimulus timing information to the ax        
    def add_stimulus_template(self,data,times=np.array([0]),start_time=0.0,source='dome',name='visual stimulus - generic',**kwargs):
        abstract = self.create_timeseries("AbstractFeatureSeries",name, "template")
        
        if 'features' in kwargs.keys():
            if 'features_units' in kwargs.keys():
                abstract.set_features(kwargs['features'],kwargs['features_units'])
            else:
                self.fatal_error("features_units is required when features is used")
        abstract.set_data(data)
        abstract.set_source(source)
        abstract.set_time(times+start_time)
        abstract.finalize()
    
    def add_eyedata(self,data,times,name='righteye',start_time=0.0):
        self.eye_mod = self.create_module(name)
        pupil_area = self.eye_mod.create_interface("PupilTracking")
        pupil_area_ts = self.create_timeseries('TimeSeries','PupilArea')
        pupil_area_ts.set_data(data['pupil_area'],conversion=np.array([1]).astype(np.float32)[0],resolution=np.array([1]).astype(np.float32)[0],unit='pixels')
        pupil_area_ts.set_time(times+start_time)
        pupil_area.add_timeseries(pupil_area_ts)
        pupil_area_ts.finalize()
        pupil_area.finalize()
        eye_position = self.eye_mod.create_interface("EyeTracking")
        eye_position_ts = self.create_timeseries('SpatialSeries','EyePositions')
        eye_position_ts.set_data(data['pupil_positions'],conversion=np.array([1]).astype(np.float32)[0],resolution=np.array([1]).astype(np.float32)[0],unit='pixels')
        eye_position_ts.set_value('reference_frame',['[0,0]'])
        eye_position_ts.set_time(times+start_time)
        eye_position.add_timeseries(eye_position_ts)
        eye_position_ts.finalize()
        eye_position.finalize()
        
        self.eye_mod.finalize()
        
    def add_to_general(self,data,name='general'):
        self.set_metadata(name,data)
        