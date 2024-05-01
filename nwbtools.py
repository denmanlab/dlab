import numpy as np
import pandas as pd
import warnings
import glob, os, h5py, csv
from dlab.generalephys import option234_positions
from dlab.sglx_analysis import readAPMeta
from dlab.utils import get_peak_waveform_from_template
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

def df_from_phy(path,site_positions = option234_positions,**kwargs):
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
    pandas DataFrame
		returns a DataFrame of 'good' units, each of which includes:
	"""
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
	"""DEPRECATED
	
	"""
	if probe_name == None: probe_name = recording_path
    #Get individual folders for each probe
	unit_times=[]
	if aligned == False:
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

def load_unit_data_from_phy(recording_path,chanmap=None,insertion_depth = 3840,insertion_angle = 0):	
	"""requires that phy has been run to generate cluster_info.tsv
	   searches the folder for the chanmap the KS used, or searches one folder up for it

    Parameters
    ----------
    recording_path : string
        the path to the sorted data
    chanmap : np.array, optional
        the geometry of the sites on the array. n x 2, where n is the number of channels. the site_positions should contain coordinates of the channels in probe space. for example, in um on the face of the probe
    insertion_depth : int, optional
        the depth in microns of the insertion
    insertion_depth : int, optional
        the angle, away from normal to the brain surface, of the insertion. used in calculating depth from puea

    Returns
    -------
    cluster_info : dict
		returns a dictionary of 'good' units
	""" 
	cluster_info = pd.read_csv(os.path.join(recording_path, 'cluster_info.tsv'), '\t')
	if cluster_info.keys()[0]=='cluster_id':
		cluster_info = cluster_info.rename(columns={'cluster_id':'id'})
	spike_clusters = np.ndarray.flatten(np.load(os.path.join(recording_path, 'spike_clusters.npy')))
	spike_templates = np.load(open(os.path.join(recording_path,'spike_templates.npy'),'rb'))
	templates = np.load(open(os.path.join(recording_path,'templates.npy'),'rb'))
	spike_times = np.load(open(os.path.join(recording_path,'spike_times.npy'),'rb'))
	timestamps = np.load(open(os.path.join(recording_path,'timestamps.npy'),'rb'))
	spike_secs = timestamps[spike_times.flatten()]

    #parse spike times for each unit. also get the template so we can use it for waveform shape clustering
	times = []
	mean_templates = []
	for unitID in cluster_info.id.values:
		times.append(spike_secs[spike_clusters == unitID])

		all_templates = spike_templates[np.where(spike_clusters==unitID)].flatten()
		if len(all_templates) > 100:
			n_templates_to_subsample = 100
		else: n_templates_to_subsample = len(all_templates)
		random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
		mean_template = np.mean(random_subsample_of_templates,axis=0)
		mean_templates.append(mean_template)
	cluster_info['times'] = times
	cluster_info['template'] = mean_templates
	cluster_info['depth_from_pia']=cluster_info.depth.values * -1 + insertion_depth*np.cos(np.deg2rad(insertion_angle))

	if chanmap == None:
		try:
			chanmap = loadmat(glob.glob(os.path.join(recording_path,'*hanMap.mat'))[0])
		except:
			chanmap = loadmat(glob.glob(os.path.join(os.path.dirname(recording_path),'*hanMap.mat'))[0])

	cluster_info['ycoords'] = chanmap['ycoords'].flatten()[cluster_info.ch.values]
	cluster_info['xcoords'] = chanmap['xcoords'].flatten()[cluster_info.ch.values]
	cluster_info['shank'] = np.floor(cluster_info['xcoords'].values / 205.).astype(int)

	return cluster_info

def make_spike_secs(probe_folder):
	"""if a times of spikes, in seconds, have not been calculated (only samples), creates spike_secs.npy

    Parameters
    ----------
    probe_folder : string
        the path to the  folder containing sorted data and raw data

    Returns
    -------
     None
	 saves, the probe_folder input, a new file called spike)secs.npy
	""" 

	c = np.load(os.path.join(probe_folder,'spike_times.npy'))
	try:
		a = np.load(os.path.join(probe_folder,'timestamps.npy'))
	except:
		try:
			a = np.load(os.path.join(probe_folder,'new_timestamps','timestamps.npy'))
		except: 
			try:
				print('could not find timestamps.npy, trying to recreate from the sync TTLs for '+probe_folder)
				ct.recreate_probe_timestamps_from_TTL(probe_folder)
				a = np.load(os.path.join(probe_folder,'new_timestamps','timestamps.npy'))
			except: print('could not find timestamps.npy')
	try:
		spike_secs = a[c.flatten()[np.where(c.flatten()<a.shape[0])]]
	except: 
		print(np.shape(a))
		print(np.shape(c.flatten()))
		print(np.shape(c))
		print('shape of spike times annd timestamps not compatible, check above and investigate.')
	np.save(open(os.path.join(probe_folder,'spike_secs.npy'),'wb'),spike_secs)

def multi_load_unit_data(recording_folder,probe_names=['A','B','C','D'],probe_depths=[3840,3840,3840,3840],spikes_filename = 'spike_secs.npy', aligned=True):
	"""requires that phy has been run to generate cluster_info.tsv
	   searches the folder for the chanmap the KS used, or searches one folder up for it

    Parameters
    ----------
    recording_folder : string
        the path to the parent folder containing multiple simultaneous recordings. each folder contains sorted data
	probe_names : tuple-like, containing strings
        the names of the probes in the recording folder
	probe_depths : tuple-like, containing ints
        the depths of insertion of the probes in the recording folder
    spikes_filename : string, optional
        the name of the file containg times for each spike. default: 'spike_secs.npy'
    aligned : bool, optional
        whether the probes are temporally aligned. default True

    Returns
    -------
     : pandas DataFrame
		a DataFrame containing good units from all recordings. also adds a column for probe of origin based on the `probe_names` input
	""" 

	folder_paths = glob.glob(os.path.join(recording_folder,'*imec*'))
	if len(folder_paths) > 0: spikes_filename = 'spike_secs.npy'
	else:
		folder_paths = glob.glob(os.path.join(recording_folder,'*AP*'))
		if len(folder_paths) > 0: 
			for probe_folder in folder_paths: make_spike_secs(probe_folder)
		else:
			print('did not find any recordings in '+recording_folder+'')
			return
	return pd.concat([load_unit_data(folder,probe_name=probe_names[i],probe_depth=probe_depths[i],spikes_filename = spikes_filename, aligned=True,df=True) for i,folder in enumerate(folder_paths)],ignore_index=True)
    
