from dlab.generalephys import option234_positions
import numpy as np
import os
def load_phy_template(path,site_positions = option234_positions,**kwargs):
# load spike data that has been manually sorted with the phy-template GUI
# the site_positions should contain coordinates of the channels in probe space. for example, in um on the face of the probe
# returns a dictionary of 'good' units, each of which includes:
#	times: spike times, in seconds
#	template: template used for matching
#	ypos: y position on the probe, calculated from the template. requires an accurate site_positions. averages template from 100 spikes.
#	xpos: x position on the probe, calcualted from the template. requires an accurate site_positions. averages template from 100 spikes.
	clusters = np.load(open(os.path.join(path,'spike_clusters.npy')))
	spikes = np.load(open(os.path.join(path,'spike_times.npy')))
	spike_templates = np.load(open(os.path.join(path,'spike_templates.npy')))
	templates = np.load(open(os.path.join(path,'templates.npy')))
	cluster_id = [];
	[cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))];
	if 'sampling_rate' in kwargs.keys():
		samplingrate = kwargs['sampling_rate']
	else:
		samplingrate =30000.
		print('no sampling rate specified, using default of 30kHz')
		
	units = {}
	for i in np.arange(1,np.shape(cluster_id)[0]):
		if cluster_id[i][0].split('\t')[1] == 'good' :#:or cluster_id[i][0].split('\t')[1] == 'unsorted' :#if it is a 'good' cluster by manual sort
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
			low_values_indices = weights < 0.25  # Where values are low,
			weights[low_values_indices] = 0      # make the weight 0
			(xpos,ypos)=np.average(site_positions,axis=0,weights=weights)
			units[str(unit)]['waveform_weights'] = weights
			units[str(unit)]['xpos'] = xpos
			units[str(unit)]['ypos'] = ypos #- site_positions[-1][1]
	return units
