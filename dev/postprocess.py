import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from glob import glob
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
        
        self.option234_xpositions           = np.zeros((192,2))
        self.option234_ypositions           = np.zeros((192,2))
        self.option234_positions            = np.zeros((384,2))
        self.option234_positions[:,0][::4]  = 21
        self.option234_positions[:,0][1::4] = 53
        self.option234_positions[:,0][2::4] = 5
        self.option234_positions[:,0][3::4] = 37
        self.option234_positions[:,1]       = np.floor(np.linspace(383,0,384)/2) * 20
        
        if 'recording' not in os.path.basename(recording_path):
            print('Please provide a path to a recording folder (e.g. /path/to/recording1)')
            
        self.recording_path = recording_path
        self.ap_folders     = glob(os.path.join(self.recording_path,'continuous','*AP'))
        
    def load(self,probe_depths=[],probes=[],acq='OpenEphys'):
        site_positions = self.option234_positions
        
        for i,PROBE in enumerate(probes):
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
                        except: print('could not load timestamps.npy')
                        spike_times = ts[spike_times]
                        np.save('spike_seconds.npy',spike_times)
                        
                    else: print('SpikeGLX currently not supported')
                    
            try:
                cluster_info = pd.read_csv('cluster_info.tsv', delimiter='\t')
            except: 
                print('Unable to load cluster_info.tsv. Have you opened this data in Phy?')
                        
            spike_clusters  = np.load('spike_clusters.npy').flatten()
            spike_templates = np.load('spike_templates.npy')
            templates       = np.load('templates.npy')
            amplitudes      = np.load('amplitudes.npy')
            weights         = np.zeros(site_positions.shape)
            
            mean_templates = []
            xpos           = []
            ypos           = []
            all_weights    = []
            amps           = []
            times          = []
            
            
            for unit_id in cluster_info.cluster_id.values:
                #get mean template for each unit
                all_templates    = spike_templates[spike_clusters==unit_id].flatten()
                random_subsample = templates[all_templates[np.random.randint(0,len(all_templates),100)]] #sample 100 templates
                mean_template    = np.mean(random_subsample,axis=0)
                mean_templates.append(mean_template)
                
                #Take weighted average of site positions where hweights is abs value of template for that channel
                #This gets us the x and y positions of the unit on the probe
                for channel in range(len(mean_template.T)):
                    weights[channel,:] = np.trapz(np.abs(mean_template.T[channel]))
                    
                weights                /= weights.max()
                weights[weights < 0.25] = 0 #Where weights are low, set to 0
                x,y                     = np.average(site_positions,weights=weights,axis=0)
                all_weights.append(weights)
                xpos.append(x)
                ypos.append(y)
                
                amps.append(amplitudes[:,0][spike_clusters==unit_id])
                times.append(spike_times[spike_clusters==unit_id])
            
            
            probe_data = cluster_info.copy()
            
            probe_data.insert(1,'times',times)
            probe_data.insert(1,'amplitudes',amps)
            probe_data.insert(1,'weights',all_weights)
            probe_data.insert(1,'template',mean_templates)
            probe_data.insert(1,'ypos',ypos)
            probe_data.insert(1,'xpos',xpos)
            probe_data.insert(0,'probe',[probe_name]*len(probe_data))
            probe_data['depth'] = np.array(ypos)-3840+probe_depths[i]
            
            
            if 'unit_data' not in locals():
                unit_data = probe_data
            else:
                unit_data.append(probe_data)
                
            os.chdir(self.old_cwd)
            
        # unit_data[len(unit_data.times) > 0]
        return(unit_data)
    
    def get_qMetrics(self,path):
        metrics_path = os.path.join(path,'qMetrics')
        if not os.path.isdir(metrics_path):
            print('Please provide path containing qMetrics folder')
        else:
            params = pd.read_parquet(os.path.join(metrics_path,'_bc_parameters._bc_qMetrics.parquet'))
            
            qMetrics = pd.read_parquet(os.path.join(metrics_path,'templates._bc_qMetrics.parquet'))
            
        return params, qMetrics
    
    def qMetrics_labels(self, probes=[]):
        
        labels = []
                    
        for i,PROBE in enumerate(probes):
            for folder in self.ap_folders:
                if 'Probe'+PROBE in folder:
                    probe_path = folder
            os.chdir(probe_path)
            
            param, qMetric = self.get_qMetrics(probe_path)
            unit_type      = np.full((len(qMetric),len(probes)),np.nan)
            
            # Noise Cluster Condtions
            b1  = qMetric.nPeaks > param.maxNPeaks[0]
            b2  = qMetric.nTroughs > param.maxNTroughs[0]
            b3  = qMetric.spatialDecaySlope >= param.minSpatialDecaySlope[0]
            b4  = qMetric.waveformDuration_peakTrough < param.minWvDuration[0]
            b5  = qMetric.waveformDuration_peakTrough > param.maxWvDuration[0]
            b6  = qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction[0]
            
            unit_type[b1 | b2 | b3 | b4 | b5 | b6] = 0 #NOISE
            
            # Somatic Cluster Conditions
            b7  = qMetric.isSomatic != param.somatic[0]
            
            unit_type[b7 & np.isnan(unit_type).T]    = 3 #NON-SOMATIC
            
            b8  = qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing[0]
            b9  = qMetric.nSpikes > param.minNumSpikes[0]
            b10 = qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations[0]
            b11 = qMetric.presenceRatio >= param.minPresenceRatio[0] 
            
            unit_type[b8 & b9 & b10 & b11 & np.isnan(unit_type).T] = 1 #GOOD
        
            if param.computeDistanceMetrics[0] > 0 :
                b12 = qMetric.Lratio <= param.lratioMax[0]
                b13 = unit_type == 1
                b13 = np.squeeze(b13)
                
                unit_type[b12 & b13] = 10 
                unit_type[unit_type == 1]      = np.nan
                unit_type[unit_type == 10]     = 1 #GOOD
                
            if param.computeDrift[0] > 0:
                b13 = unit_type == 1
                b13 = np.squeeze(b13)
                b14 = qMetric.rawAmplitude > param.minAmplitude[0]
                b15 = qMetric.maxDriftEstimate <= param.maxDrift[0]
                
                unit_type[b13 & b14 & b15] = 10
                unit_type[unit_type == 1]  = np.nan
                unit_type[unit_type == 10] = 1 #GOOD
            
            if param.extractRaw[0] > 0:
                b13 = unit_type == 1
                b13 = np.squeeze(b13)
                b14 = qMetric.rawAmplitude > param.minAmplitude[0]
                b16 = qMetric.signalToNoiseRatio >= param.minSNR[0]
                
                unit_type[b13 & b14 & b16] = 10
                unit_type[unit_type == 1]      = np.nan
                unit_type[unit_type == 10]     = 1 #GOOD
                
            if pd.notna(param.isoDmin[0]):
                if 'isoD' in qMetric:
                    b13 = unit_type == 1
                    b13 = np.squeeze(b13)
                    b17 = qMetric.isoD >= param.isoDmin[0]
                    
                    unit_type[b13 & b17] = 10
                    unit_type[unit_type == 1]      = np.nan
                    unit_type[unit_type == 10]     = 1 #GOOD
                                
            unit_type[np.isnan(unit_type)] = 2 #MUA
            

            for i in unit_type:
                if i == 0:
                    labels.append('noise')
                if i == 1:
                    labels.append('good')
                if i == 2:
                    labels.append('mua')
                if i == 3:
                    labels.append('non-somatic')
           
        return labels
    
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
    
    def get_dlines(self):
        return self.digital_output
    
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