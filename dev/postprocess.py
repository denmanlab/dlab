import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from random import sample
import os
from glob import glob
from scipy.io import loadmat
try:
    from OpenEphys.analysis import Session
except ImportError:
    try:
        from open_ephys.analysis import Session
    except ImportError:
        print('no OpenEphys.py get this from https://github.com/open-ephys/analysis-tools')

warnings.simplefilter(action='ignore', category=FutureWarning)


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
                            print('could not load timestamps.npy, calculating from synch signals')
                            events_path = os.path.join(glob(os.path.join(self.recording_path,'events','*Probe'+PROBE+'*'))[0],'TTL')
                            spike_samples = np.load('spike_times.npy').flatten()
                            samples = np.load(os.path.join(events_path,'sample_numbers.npy')).flatten()
                            sample_times = np.load(os.path.join(events_path,'timestamps.npy')).flatten()
                            start_sample = samples[0]
                            spike_times = np.array([sample_times[np.where(samples < int(s + start_sample))[0][0]]  + ((int(s + start_sample)) - samples[np.where(samples < int(s + start_sample))[0][0]] )/30000. for s in spike_samples])
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
    
class StimData:
    def __init__(self,recording_path, recording_no = 0) -> None:
        self.old_cwd       = os.getcwd()
        self.sampling_rate = 30000.0
        
        if 'recording' not in os.path.basename(recording_path):
            print('Please provide a path to a recording folder (e.g. /path/to/recording1)')
            
        self.session_path = os.path.abspath(os.path.join(recording_path,*[os.pardir]*3))
        self.session      = Session(self.session_path)
        self.recording    = self.session.recordnodes[0].recordings[recording_no-1]
        
        events            = self.recording.events #DataFrame
        
        stim_events       = events[events.stream_index == events.stream_index.unique().max()]
        self.stim_rising  = stim_events[stim_events.state == 1]
        self.stim_falling = stim_events[stim_events.state == 0]
        
        self.digital_output = {}
        self.dlines         = np.sort(events.line.unique())
        
        for dl in self.dlines:
            self.digital_output[str(dl-1)] = self.stim_rising[self.stim_rising.line == dl].timestamp.values
    
    def plot(self):
        fig,ax = plt.subplots()
        for dline in self.digital_output:
            ax.plot(self.digital_output[dline],np.ones(len(self.digital_output[dline]))*int(dline),'-o',label=dline)
        plt.title('Digital Events')
        plt.ylabel('Digital Line')
        plt.xlabel('Time (s)')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        
