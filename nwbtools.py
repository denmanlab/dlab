import numpy as np
import pandas as pd
import warnings
import glob, os, h5py, csv
from dlab.generalephys import option234_positions
from dlab.sglx_analysis import readAPMeta
import dlab.continuous_traces as ct

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
		
def load_phy_template(path,cluster_file='KS2',site_positions = option234_positions, **kwargs):
	"""load spike data that has been manually sorted with the phy-template GUI

    Parameters
    ----------
    path : string
        the path to the sorted data
    cluster_file : string, optional
        the format of the cluter_info file. options, KS2, KS3
    site_positions : np.array, optional
        the geometry of the sites on the array. n x 2, where n is the number of channels. the site_positions should contain coordinates of the channels in probe space. for example, in um on the face of the probe

    Returns
    -------
    dict
		returns a dictionary of 'good' units, each of which includes:
			times: spike times, in seconds
			template: template used for matching
			ypos: y position on the probe, calculated from the template. requires an accurate site_positions. averages template from 100 spikes.
			xpos: x position on the probe, calcualted from the template. requires an accurate site_positions. averages template from 100 spikes.
	""" 
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

class UnitData:
    def __init__(self,recording_path) -> None:
        self.old_cwd       = os.getcwd()
        self.sampling_rate = 30000.0
        
        if 'recording' not in os.path.basename(recording_path):
            print('Please provide a path to a recording folder (e.g. /path/to/recording1)')
            
        self.recording_path = recording_path
        self.all_folders    = glob(os.path.join(self.recording_path,'continuous','*'))
        self.ap_folders     = [folder for folder in self.all_folders if 'LFP' not in os.path.basename(folder)]
        self.ap_folders     = [folder for folder in self.ap_folders if 'NI-DAQmx' not in os.path.basename(folder)]
        
    def load(self,probe_depths=[],probes=[],acq='OpenEphys',ignore_phy=False):
        # site_positions = self.option234_positions
        
        for i,PROBE in enumerate(probes):
            print(PROBE)
            probe_name = PROBE
            for folder in self.ap_folders:
                if 'Probe'+PROBE in folder:
                    probe_path = folder
            
            os.chdir(probe_path)
            
            if os.path.isfile('spike_seconds.npy'):
                spike_times = np.load('spike_seconds.npy')
            else:
                spike_times = np.load('spike_times.npy').flatten()
                if isinstance(spike_times[0],(np.uint64)):
                    if acq == 'OpenEphys':
                        try:
                            ts = np.load('timestamps.npy')
                            spike_times = ts[spike_times]
                            np.save('spike_seconds.npy',spike_times)
                        
                        except: 
                            print('could not load timestamps.npy')
                            spike_times = spike_times/self.sampling_rate
                            np.save('spike_seconds.npy',spike_times)
                            

                    else: print('SpikeGLX currently not supported')
                    
                    
            site_positions  = np.load('channel_positions.npy')               
            spike_clusters  = np.load('spike_clusters.npy').flatten()
            spike_templates = np.load('spike_templates.npy')
            templates       = np.load('templates.npy')
            amplitudes      = np.load('amplitudes.npy')
            
            cluster_info = None        
            try:
                cluster_info = pd.read_csv('cluster_info.tsv', delimiter='\t')
                if ignore_phy == True:
                    cluster_info    = None
                    cluster_Amps    = pd.read_csv('cluster_Amplitude.tsv', delimiter='\t')
                    ContamPct       = pd.read_csv('cluster_ContamPct.tsv', delimiter='\t')
                    KSLabel         = pd.read_csv('cluster_KSLabel.tsv', delimiter='\t')
            except: 
                print('Unable to load cluster_info.tsv. Have you opened this data in Phy?')
                cluster_Amps    = pd.read_csv('cluster_Amplitude.tsv', delimiter='\t')
                ContamPct       = pd.read_csv('cluster_ContamPct.tsv', delimiter='\t')
                KSLabel         = pd.read_csv('cluster_KSLabel.tsv', delimiter='\t')

            
            weights        = np.zeros(site_positions.shape)
            mean_templates = []
            peak_templates = []

            all_weights    = []
            amps           = []
            times          = []
            ch             = []
            
            
            for unit_id in np.unique(spike_clusters):
                #get mean template for each unit
                all_templates,count    = np.unique(spike_templates[np.where(spike_clusters==unit_id)],return_counts=True)
                
                if len(all_templates) > 100:
                    n_templates_to_subsample = 100
                else: 
                    n_templates_to_subsample = len(all_templates)
                
                random_subsample_of_templates = templates[sample(list(all_templates),n_templates_to_subsample)]
                
                mean_template = np.mean(random_subsample_of_templates,axis=0)
                
                mean_templates.append(mean_template)
                
                if cluster_info is not None:
                    best_ch = cluster_info[cluster_info.cluster_id == unit_id].ch.values[0].astype(int)
                else:
                    best_ch = np.argmax((np.max(mean_template,axis=0) - np.min(mean_template,axis=0)))
                
                ch.append(best_ch)
                
                peak_wv = mean_template[:,best_ch-1]
                peak_templates.append(peak_wv)
                
                #Take weighted average of site positions where hweights is abs value of template for that channel
                #This gets us the x and y positions of the unit on the probe

                # for channel in range(len(mean_template.T)):
                #     weights[channel,:] = np.trapz(np.abs(mean_template.T[channel]))
            
                # # weights                /= weights.max()
                # weights[weights < 0.25] = 0 #Where weights are low, set to 0
                # x,y                     = np.average(site_positions,weights=weights,axis=0)
                # all_weights.append(weights)
                # xpos.append(x)
                # ypos.append(y)
                
                amps.append(amplitudes[:,0][spike_clusters==unit_id])
                times.append(spike_times[spike_clusters==unit_id])

            if cluster_info is not None:
                probe_data = cluster_info.copy()
            else:
                probe_data = pd.DataFrame()
                probe_data['cluster_id'] = np.unique(spike_clusters)
                probe_data['Amplitude']  = cluster_Amps[np.in1d(cluster_Amps.cluster_id.values,np.unique(spike_clusters))]['Amplitude'].values 
                probe_data['ContamPct']  = ContamPct[np.in1d(ContamPct.cluster_id.values,np.unique(spike_clusters))]['ContamPct'].values 
                probe_data['KSLabel']    = KSLabel[np.in1d(KSLabel.cluster_id.values,np.unique(spike_clusters))]['KSLabel'].values 
                probe_data['ch']         = ch
                
            probe_data['probe']      = [probe_name]*len(probe_data)
            # probe_data['shank']    = np.floor(cluster_info['xcoords'].values / 205.).astype(int)
            probe_data['depth']      = np.array(site_positions[:,1][ch])*-1 + probe_depths[i]
            probe_data['times']      = times
            probe_data['amplitudes'] = amps
            probe_data['template']   = mean_templates
            # probe_data['weights']  = all_weights
            probe_data['peak_wv']    = peak_templates
            probe_data['xpos']       = site_positions[:,0][probe_data['ch']]
            probe_data['ypos']       = site_positions[:,1][probe_data['ch']]
            probe_data['n_spikes']   = [len(i) for i in times]
            
            if 'unit_data' not in locals():
                unit_data = probe_data
                
            else:
                unit_data = pd.concat([unit_data,probe_data],ignore_index=True)
                unit_data.reset_index(inplace=True,drop=True)
                
            os.chdir(self.old_cwd)
        
        return unit_data
    
    def get_qMetrics(self,path):
        metrics_path = os.path.join(path,'qMetrics')
        if not os.path.isdir(metrics_path):
            print('Please provide path containing qMetrics folder')
        else:
            params = pd.read_parquet(os.path.join(metrics_path,'_bc_parameters._bc_qMetrics.parquet'))
            
            qMetrics = pd.read_parquet(os.path.join(metrics_path,'templates._bc_qMetrics.parquet'))
            
        return params, qMetrics
    
    def qMetrics_labels(self, probes=[],param_changes = {}):
        ids = []
        all_labels = []            
        for i,PROBE in enumerate(probes):
            labels = []
            for folder in self.ap_folders:
                if 'Probe'+PROBE in folder:
                    probe_path = folder
            
            os.chdir(probe_path)
            
            param, qMetric = self.get_qMetrics(probe_path)
            cluster_id     = qMetric.phy_clusterID.values.astype(int)
            unit_type      = np.full((len(qMetric)),np.nan)
            
            if param_changes:
                for key in param_changes.keys():
                    param[key] = param_changes[key]
            
            # Noise Cluster Condtions
            noise0  = pd.isnull(qMetric.nPeaks)
            noise1  = qMetric.nPeaks                      > param.maxNPeaks[0]
            noise2  = qMetric.nTroughs                    > param.maxNTroughs[0]
            noise3  = qMetric.spatialDecaySlope           > param.minSpatialDecaySlope[0]
            noise4  = qMetric.waveformDuration_peakTrough < param.minWvDuration[0]
            noise5  = qMetric.waveformDuration_peakTrough > param.maxWvDuration[0]
            noise6  = qMetric.waveformBaselineFlatness    > param.maxWvBaselineFraction[0]
            
            unit_type[noise0 | noise1 | noise2 | noise3 | noise4 | noise5 | noise6] = 0 #NOISE
            
            #MUA Conditions
            mua0 = qMetric.percentageSpikesMissing_gaussian > param.maxPercSpikesMissing[0]
            mua1 = qMetric.nSpikes                          < param.minNumSpikes[0]
            mua2 = qMetric.fractionRPVs_estimatedTauR       > param.maxRPVviolations[0]
            mua3 = qMetric.presenceRatio                    < param.minPresenceRatio[0]
            
            unit_type[(mua0| mua1 | mua2 | mua3)&np.isnan(unit_type)] = 2 #MUA
            
            #Optional MUA metrics
            if param.computeDistanceMetrics[0] == 1 & ~param.isoDmin.isna()[0]:
                mua4 = qMetric.isoD   < param.isoDmin[0]
                mua5 = qMetric.Lratio > param.lratioMax[0]
                
                unit_type[(mua4 | mua5)&np.isnan(unit_type)] = 2 #MUA

            else:
                print('No distance metrics calculated')
                
            if param.extractRaw[0] == 1:
                mua6 = qMetric.rawAmplitude       < param.minAmplitude[0]
                mua7 = qMetric.signalToNoiseRatio < param.minSNR[0]
                unit_type[(mua6 | mua7)&np.isnan(unit_type)] = 2 #MUA

            else:
                print('Raw waveforms not extracted')
                                    
            # Somatic Cluster Conditions
            if param.splitGoodAndMua_NonSomatic[0]:
                nsom0 = qMetric.isSomatic != param.somatic[0]
                unit_type[(unit_type==1) & (nsom0)] = 3 #Good Non-Somatic
                unit_type[(unit_type==2) & (nsom0)] = 4 #MUA Non-Somatic
            
            #GOOD Conditions
            unit_type[np.isnan(unit_type)] = 1 #Good

            for i in unit_type:
                if i == 0:
                    labels.append('NOISE')
                    all_labels.append('NOISE')
                if i == 1:
                    labels.append('GOOD')
                    all_labels.append('GOOD')
                if i == 2:
                    labels.append('MUA')
                    all_labels.append('MUA')
                if i == 3:
                    labels.append('NON-SOMA GOOD')
                    all_labels.append('NON-SOMA GOOD')
                if i == 4:
                    labels.append('NON-SOMA MUA')
                    all_labels.append('NON-SOMA MUA')
                    
            ids += list(cluster_id)
          
            out_df = pd.DataFrame({'cluster_id':cluster_id, 'bc_unitType':labels})
        # if os.path.isfile('cluster_bc_unitType.tsv'):
        #     q = input('Would you like to overwrite cluster_bc_unitType.tsv? (Y/N)')
        #     if q == 'Y':
        #         print('Overwriting  cluster_bc_unitType.tsv')
        #         out_df.to_csv('cluster_bc_unitType.tsv', sep='\t', index=False, header=True)
        #     if q == 'N':
        #         print('No output saved')
            print('Saving output....')
            out_df.to_csv('cluster_bc_unitType.tsv', sep='\t', index=False, header=True)
                
        return {'cluster_id':ids, 'qm_labels':all_labels}
    
