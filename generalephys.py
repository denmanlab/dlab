# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:01:49 2014

@author: danieljdenman
"""

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.optimize as opt
from scipy.signal import resample
from scipy.stats import linregress
from scipy.stats import ttest_ind
from scipy.ndimage import zoom
import os, csv
import pandas as pd
#from skimage import transform

import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection


# import pyqtgraph as pg
# import pyqtgraph.opengl as gl
#from PyQt4 import QtGui, QtCore

#from pykCSD.pykCSD import KCSD
# import djd.electrode_maps as maps
# import djd.OpenEphys as oe

try:
    import phy
    from phy.session import Session as physesssion
except:
    'no phy install available, phy-based tools will not work' 

#defines the global sampling rate.
#this is used throughout to convert from sample number to time.
#for ecube, headstage sampling rate is 25kHz.
#adjust for other acquisition systems accordingly.
samplingRate=30000.0

option234_xpositions = np.zeros((192,2))
option234_ypositions = np.zeros((192,2))
option234_positions = np.zeros((384,2))
option234_positions[:,0][::4] = 21
option234_positions[:,0][1::4] = 53
option234_positions[:,0][2::4] = 5
option234_positions[:,0][3::4] = 37
option234_positions[:,1] = np.floor(np.linspace(383,0,384)/2) * 20
# imecp3_image = plt.imread(os.path.join(os.path.dirname(os.path.abspath(maps.__file__)),'imec_p3.png'))

imec_p2_positions = np.zeros((128,2))
imec_p2_positions[:,0][::2] = 18
imec_p2_positions[:,0][1::2] = 48
imec_p2_positions[:,1] = np.floor(np.linspace(0,128,128)/2) * 20;imec_p2_positions[:,1][-1]=1260.
# imecp2_image = plt.imread(os.path.join(os.path.dirname(os.path.abspath(maps.__file__)),'imec_p2.png'))

#a set of 50 maximally distinct colors, from http://tools.medialab.sciences-po.fr/iwanthue/
color50 = ["#67572e",
"#272d38",
"#e47689",
"#7499db",
"#64e251",
"#4553a8",
"#e55728",
"#325338",
"#da94d7",
"#591d42",
"#d3e571",
"#c7af44",
"#5e3ecc",
"#7f75df",
"#9d5258",
"#a3e841",
"#cb48b4",
"#4a6890",
"#c3e19a",
"#77606d",
"#50a874",
"#e1e53b",
"#68e0ca",
"#ac2b51",
"#cf9894",
"#829b44",
"#e54150",
"#da4687",
"#382f1c",
"#927933",
"#73c5dc",
"#dc865f",
"#925991",
"#e8b12e",
"#b22d25",
"#518c8e",
"#3d6e2a",
"#572282",
"#55ad3d",
"#cf832e",
"#8a9675",
"#dabd88",
"#62221e",
"#6fe594",
"#9ab92f",
"#312557",
"#b74cdf",
"#994923",
"#c1b4d1",
"#c5dac7"]


#=================================================================================================
#--------loading, clustering, and waveform shape-related utilites-------------------------------------
#=================================================================================================
def load_phy_kwik(filepath,**kwargs):
#load spike data that has been manually sorted with the phy GUI
#returns a phy.Session that includes:
#	session.store.MUA: list of MUA cluster IDs
#	session.store.good: list of 'good' cluster IDs
#	session.store.times: dictionary of spike time arrays, keyed by cluster ID
	session = Session(filepath)
	print('...parsing spike times...')
	
	times={}
	good = []
	MUA = []
	
	for i,cluster_group in enumerate(session.store.model.cluster_groups.values()):
		if cluster_group == 1:
			cluster_number = session.store.model.cluster_ids[i]
			MUA.append(cluster_number)
			times[cluster_number]=np.array([session.store.model.spike_samples[t] for t in np.where(session.store.model.spike_clusters==cluster_number)[0]])
		if cluster_group ==2:
			cluster_number = session.store.model.cluster_ids[i]
			good.append(cluster_number)
			times[cluster_number]=np.array([session.store.model.spike_samples[t] for t in np.where(session.store.model.spike_clusters==cluster_number)[0]])	
	
	session.store.MUA = MUA
	session.store.good = good	
	session.store.times = times
	return session

# def load_phy_template(path,cluster_file='KS2',site_positions = option234_positions,**kwargs):
# # load spike data that has been manually sorted with the phy-template GUI
# # the site_positions should contain coordinates of the channels in probe space. for example, in um on the face of the probe
# # returns a dictionary of 'good' units, each of which includes:
# #	times: spike times, in seconds
# #	template: template used for matching
# #	ypos: y position on the probe, calculated from the template. requires an accurate site_positions. averages template from 100 spikes.
# #	xpos: x position on the probe, calcualted from the template. requires an accurate site_positions. averages template from 100 spikes.
#     clusters = np.load(open(os.path.join(path,'spike_clusters.npy'),'rb'))
#     spikes = np.load(open(os.path.join(path,'spike_times.npy'),'rb'))
#     spike_templates = np.load(open(os.path.join(path,'spike_templates.npy'),'rb'))
#     templates = np.load(open(os.path.join(path,'templates.npy'),'rb'))
#     cluster_id = []
#     if cluster_file == 'KS2':
#         [cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_KSLabel.tsv')))]
#     else:
#         if os.path.isfile(os.path.join(path,'cluster_group.tsv')):
#             # cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_group.tsv')))][1:]
#             [cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_group.tsv')))]
#         else:
#             if os.path.isfile(os.path.join(path,'cluster_groups.csv')):
#                 # cluster_id = [row for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))][1:]
#                 [cluster_id.append(row) for row in csv.reader(open(os.path.join(path,'cluster_groups.csv')))]
#             else: print('cant find cluster groups, either .tsv or .csv')
#     if 'sampling_rate' in kwargs.keys():
#         samplingrate = kwargs['sampling_rate']
#     else:
#         samplingrate =30000.
#         # print('no sampling rate specified, using default of 30kHz')
        
#     units = {}
#     for i in np.arange(1,np.shape(cluster_id)[0]):
#         if cluster_id[i][0].split('\t')[1] == 'good' :#:or cluster_id[i][0].split('\t')[1] == 'unsorted' :#if it is a 'good' cluster by manual sort
#             unit = int(cluster_id[i][0].split('\t')[0])
#             units[str(unit)] = {}
            
#             #get the unit spike times
#             units[str(unit)]['times'] = spikes[np.where(clusters==unit)]/samplingrate
#             units[str(unit)]['times'] = units[str(unit)]['times'].flatten()
            
#             #get the mean template used for this unit
#             all_templates = spike_templates[np.where(clusters==unit)].flatten()
#             n_templates_to_subsample = 100
#             random_subsample_of_templates = templates[all_templates[np.array(np.random.rand(n_templates_to_subsample)*all_templates.shape[0]).astype(int)]]
#             mean_template = np.mean(random_subsample_of_templates,axis=0)
#             units[str(unit)]['template'] = mean_template
            
#             #take a weighted average of the site_positions, where the weights is the absolute value of the template for that channel
#             #this gets us the x and y positions of the unit on the probe.
#             # print(mean_template.T.shape)
#             weights = np.zeros(site_positions.shape)
#             for channel in range(mean_template.T.shape[0]):
#                 weights[channel,:]=np.trapz(np.abs(mean_template.T[channel,:]))
#             weights = weights/np.max(weights)
#             low_values_indices = weights < 0.25  # Where values are low,
#             weights[low_values_indices] = 0      # make the weight 0
#             (xpos,ypos)=np.average(site_positions,axis=0,weights=weights)
#             units[str(unit)]['waveform_weights'] = weights
#             units[str(unit)]['xpos'] = xpos
#             units[str(unit)]['ypos'] = ypos #- site_positions[-1][1]
#     if 'return' in kwargs.keys():
#         if kwagrs['return']=='df':
#             return pd.DataFrame.from_dict(units,orient='index')
#     else: return units
#     else: return units

def df_from_phy(path,site_positions = option234_positions,**kwargs):
    units = load_phy_template(path,site_positions)
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

def kwik_to_csv(filepath,sampling_rate=20000.,name='units'):
	import csv
	try:
		import phy
		count=0
		session = phy.session.Session(filepath)
		units = open(os.path.join(os.path.dirname(filepath),name+'.csv'),'w')
		a = csv.writer(units)
		for i,u in enumerate(session.store.model.spikes_per_cluster.keys()):
			if session.store.model.cluster_groups[u] == 2:
				count+=1
				max_channel = np.where(session.store.mean_waveforms(u) == np.min(session.store.mean_waveforms(u)))[1][0]+1
				for spike_sample in session.store.model.spikes_per_cluster[u]:
					a.writerow([u,spike_sample/float(sampling_rate),max_channel])
		units.close()
		print(str(count)+' good units, with times and max channel, written to: '+os.path.join(os.path.dirname(filepath),name+'.csv'))

	except:
		print('doing it the old-fashioned way. no max channel data included.')
		data = ephys.oe.load_kwik_klusters(filepath)
		
		units = open(os.path.join(os.path.dirname(filepath),'units.csv'),'w')
		a = csv.writer(units) 
		for u in data['units']:
			for time in data[u]['times']:
				a.writerow([u,time/20000.])
		#a.writerows([[int(u)]+data[u]['times'] for u in data['units']])
		units.close()
		
	
#average the waveforms, by unit, in sorted data in a .kwik loaded in to a dictionary
def averagewaveforms(data):
    units = data.keys()
    numberofspikes = 0 
    for unit in units:
        if type(data[unit]) is dict:
            if 'waveforms_raw' in data[unit].keys():
                numberofspikes = np.shape(data[unit]['waveforms_raw'])[0]
                if numberofspikes > 2 :
                    waveform = np.zeros((np.shape(data[unit]['waveforms_raw'])[1],np.shape(data[unit]['waveforms_raw'])[2]))
                    for i in range(numberofspikes):
                        waveform += data[unit]['waveforms_raw'][i][:][:]
                else:
                    waveform=np.zeros((20,128))
                waveform /= numberofspikes
                waveform -= waveform[0]
                data[unit]['waveform'] = np.array(waveform) / 2.8

#find the y position of an average waveform, given the probe geometry
def findspikeprobeY(waveform,geometry):
    y = 0
    min = 0
    for row,channel in enumerate(geometry[0]):
        #print(''.join(('y: ',str(row),' ',str(geometry[0][row]-1),',',str(geometry[1][row]-1))))
        if np.min(np.concatenate((waveform[:,geometry[0][row]-1] - np.mean(waveform[:,geometry[0][row]-1]),waveform[:,geometry[1][row]-1]-np.mean(waveform[:,geometry[1][row]-1])))) < min:
            y = row
            min = np.min(np.concatenate((waveform[:,geometry[0][row]-1] - np.mean(waveform[:,geometry[0][row]-1]),waveform[:,geometry[1][row]-1]-np.mean(waveform[:,geometry[1][row]-1]))))
    return y

#make a 3-d matrix of the average waveform, dimensions being space-space-time
def getwaveformspacetime(data,geometry,yrange=120):
    nrows = max(len(p) for p in geometry)
    ncols = np.shape(geometry)[0] * 2
    ntimepoints = np.shape(data)[0]
    matrix = np.zeros((nrows,ncols, ntimepoints))   
    for column,c in enumerate(geometry):
        c = c[::-1]
        for row,channel in enumerate(c):
            waveform = data[:,channel-1]
            for timepoint in range(len(waveform)):
                matrix[row,column,timepoint] = float(waveform[timepoint])/float(yrange)
    return matrix[:,:,:]     
    
     
#find the x and y position of an average waveform, given the probe geometry
def getcentroid(waveform,geometry = np.array((np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2]))):
    im = ()
    for column,c in enumerate(geometry):
        for row,channel in enumerate(c):
            wave =  waveform[:,channel-1]
            if column == 0:
                r = wave.tolist()
                im=im+(r,)
            else:
                im[row].extend(wave.tolist())

    #plt.imshow(im)
    print(np.shape(im))
    return im#ndimage.measurements.center_of_mass(im)

#find the x and y position of an average waveform, given the probe geometry               
def findcentroid(waveform,matrix,numchannels=128):
    y = findspikeprobeY(waveform,np.array((np.linspace(1,numchannels,numchannels)[::2],np.linspace(1,numchannels,numchannels)[1::2])))
    #centroid = ndimage.measurements.center_of_mass(matrix * -1)
    im = matrix[y-2:y+2,0:2,15]
    centroid = ndimage.measurements.center_of_mass(np.abs(im))
    #return (np.abs(y+centroid[0]-64),centroid[1])
    return (y,centroid[1])

#find the x and y position of all the units in a recording, given the probe geometry
def findcentroids(data,geometry,yrange=250,numpnts=37,numchannels=128):
    units = data.keys()
    for unit in units:
        if type(data[unit]) is dict:
            if 'waveform' in data[unit].keys():
                if np.shape(data[unit]['waveform']) == (numpnts,numchannels):
                    data[unit]['centroid'] = findcentroid(data[unit]['waveform'],getwaveformspacetime(data[unit]['waveform'],geometry,yrange),numchannels=numchannels)
                else:
                    data[unit]['centroid']=(0,0)
					

#read a .kwik and return a summary of the spikes within. 
def spike_info_from_recording(folderpath,filename):
    #folderpath = os.path.dirname(filename)
    if os.path.isfile(os.path.join(folderpath,filename)):
        if len([f for f in os.listdir(folderpath) if f.endswith('.continuous')]) > 0:
            spikes = oe.load_kwik_klusters(os.path.join(folderpath,filename))
            #averagewaveforms(spikes)
            #findcentroids(spikes,np.array((np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2])))
            timeinseconds = float(np.shape(oe.load(os.path.join(folderpath,[f for f in os.listdir(folderpath) if f.endswith('.continuous')][0]))['data'])[0] / samplingRate)
                        
            output = {}
            output['no_units'] = len(spikes['units'])
            output['no_mua'] = len(spikes['mua'])
            output['totalspikes_units'] =np.sum([len(spikes[u]['times']) for u in spikes['units']])
            output['totalspikes_mua'] = np.sum([len(spikes[u]['times']) for u in spikes['mua']])
            output['avgrate_units'] = np.mean(np.array([len(spikes[u]['times']) for u in spikes['units']]))/ timeinseconds
            output['avgrate_mua'] = np.mean(np.array([len(spikes[u]['times']) for u in spikes['mua']]))/ timeinseconds
            output['rates_units'] = np.array([len(spikes[u]['times']) for u in spikes['units']])/ timeinseconds
            output['rates_mua'] = np.array([len(spikes[u]['times']) for u in spikes['mua']])/ timeinseconds
            if 'centroid' in spikes[spikes['units'][0]].keys():    
                output['position_units'] = [spikes[u]['centroid'] for u in spikes['units']]
                output['position_mua'] = [spikes[u]['centroid'] for u in spikes['mua']]
            return output
            
        else:
            print('a .continuous file must be in the folder [to get total recording time]')
    else:
        print('kwik not found in folder')


#read a list of paths to .kwik files and return a summary of the spikes within each file. 
def spike_infolist(tupleofkwikfiles):

    output = {}
    keys = ['experiment_name','no_units','no_mua','totalspikes_units', 'totalspikes_mua','avgrate_units','avgrate_mua','rates_units','rates_mua','postion_units','position_mua']
    for key in keys:
        output[key]=[]

    for filename in tupleofkwikfiles:
        print(os.path.normpath(filename))
        info = spike_info_from_recording('r'+filename)
        output['experiment_name'].append(filename)
        for key in keys:
            output[key].append(info[key])
    
    return output

#characterizes the size and spread of a waveform across a probe.
#finds the number of channels that have an average waveform amplitude above 30uV
#finds the peak-to-peak amplitude of the channel with the largest waveform
def waveform_power(data,geometry=(np.linspace(1,128,128)[::2][::-1],np.linspace(1,128,128)[1::2][::-1])):
   #set up the figure
    nrows = max(len(p) for p in geometry)
    ncols = np.shape(geometry)[0]
#    fig = plt.figure(figsize = (2,12)) # set the figure size to be square
#    gs = gridspec.GridSpec(nrows, ncols,wspace=0.4, hspace=0.02, left = 0.2, right = 0.3, bottom = 0.05, top = 0.95) 
    num_channels = 0
    biggest = 0
    amplitude = 0
    #add the data
    for column,c in enumerate(geometry):
        for row,channel in enumerate(c):
           #plot
            #ax=plt.subplot(gs[nrows*column+row])
            p2p = np.abs(np.min(data[:,channel-1]) - np.max(data[:,channel-1]))
            if p2p > 30:
                num_channels +=1
                if p2p > amplitude:
                    amplitude = p2p
     
    return (num_channels,amplitude)

#returns a list of tuples, containing unit IDs matched across two recordings
#requires the data to have been sorted with phy
def findMatchingUnits(pathToKwikOne,pathToKwikTwo):
	#load the two recordings
	data1 = physesssion(pathToKwikOne)
	data2 = physesssion(pathToKwikTwo)
	
	#make ordered lists of the units from each recording that have been called 'good'
	#the order is based on probe Y position, so that cells with similar positions on the probe can be compared.
	ordered_list_data1 = [u for u in data1.store.model.cluster_ids[data1.store.mean_probe_position(data1.store.model.cluster_ids)[:,1].argsort()][::-1] if data1.store.model.cluster_groups[u] == 2][:data1.store.model.cluster_groups.values().count(2)]
	ordered_list_data2 = [u for u in data2.store.model.cluster_ids[data2.store.mean_probe_position(data2.store.model.cluster_ids)[:,1].argsort()][::-1] if data2.store.model.cluster_groups[u] == 2][:data2.store.model.cluster_groups.values().count(2)]
	
	#find matches.
	#developed in a notebook on 192079 binary noise; this simply subtracts the mean waveforms across recordings, and calls the smallest difference the closest match.
	#if there is no close match [based on a heuristic], says there is no match. 
	list_data1 = []
	list_data2 = []
	ds = []
	for i,u in enumerate(ordered_list_data1):
		#find the subsection of channels to look for the matching waveform
		center = np.where(data1.store.mean_masks(u) == np.max(data1.store.mean_masks(u)))[0][0]
		if center - 4 < 0 :
			u_waveform = data1.store.mean_waveforms(u)[:,:center+4]
	
		else: 
			if center + 4 > len(data1.store.mean_masks(u)):
				u_waveform = data1.store.mean_waveforms(u)[:,center-4:]
			else:
				u_waveform = data1.store.mean_waveforms(u)[:,center-4:center+4]
	
		#select just the nearest [10] waveforms to test unless we're near the edges, in which case only 5 to test
		start_testunit = 0 if i - 8 < 0 else i - 8
		end_testunit = len(ordered_list_data2) if i + 8 > len(ordered_list_data2) else i + 8
		testers = ordered_list_data2[start_testunit:end_testunit] # the closest 10 channels
	
		distances= []
		for test_u in testers:
			if center - 4 < 0 :
				test_u_waveform = data2.store.mean_waveforms(test_u)[:,:center+4]
			else:
				try:
					if center + 4 > len(data1.store.mean_masks(test_u)):
						test_u_waveform = data2.store.mean_waveforms(test_u)[:,center-4:]
					else:
						test_u_waveform = data2.store.mean_waveforms(test_u)[:,center-4:center+4]
				except:
					test_u_waveform = data2.store.mean_waveforms(test_u)[:,center-4:center+4]
			if u_waveform.shape == test_u_waveform.shape:
				d=np.mean(np.abs(u_waveform - test_u_waveform))
			else:
				d=10000#what? the shapes aren't the same. this doens't make sense. so just call it a very much not match.
			distances.extend([d])
		distance = np.min(distances)
		match = testers[distances.index(distance)]
	
		#check to see if the best match is actually any good
		#if not, add this unit with no match
		if distance > 40.0:#match was no good; heuristic based on some examples and the distribution in the developmental dataset
			match = -1
			list_data1.extend([u]);list_data2.extend([None]);ds.extend([distance])
	
		else:#match was good
			if match in list_data2:#we've already found a match within the set we're testing against
				#find out which match is better, the one you just found or the one that is already found
				previous_u = list_data1[list_data2.index(match)]
				if center - 4 < 0 :
					previous_distance = np.mean(np.abs(data1.store.mean_waveforms(previous_u)[:,4:center+4] - test_u_waveform)) 
				else:
					try:
						if center + 4 > len(data1.store.mean_masks(test_u)):
							previous_distance = np.mean(np.abs(data1.store.mean_waveforms(previous_u)[:,center-4:] - test_u_waveform)) 
						else:
							previous_distance = np.mean(np.abs(data1.store.mean_waveforms(previous_u)[:,center-4:center+4] - test_u_waveform)) 
					except:
							previous_distance = np.mean(np.abs(data1.store.mean_waveforms(previous_u)[:,center-4:center+4] - test_u_waveform)) 
				if distance < previous_distance:#this match is better
					#remove the old match
					list_data1.remove(previous_u);list_data2.remove(match)
	
					#add this one
					list_data1.extend([u]);list_data2.extend([match]);ds.extend([distance])
	
				else:#the old match is better
					distances.remove(distance)#remove the best match
					testers.remove(match)#remove the best match
					if np.min(distances) < 40.0: # check if second best is any good
						list_data1.extend([u]);list_data2.extend([testers[distances.index(np.min(distances))]]);ds.extend([distance])
	
					else:#there wasn't a good match, besides the one that matches someone else better.
						list_data1.extend([u]);list_data2.extend([None]);ds.extend([distance])
	
			else: #this is the first match with the units we're testing against
				list_data1.extend([u]);list_data2.extend([match]);ds.extend([distance])

	#tack on the units from the recording tested against, with no match. 
	for test_u in ordered_list_data2:
		if test_u in list_data2:
			pass
		else:
			list_data1.extend([None]);list_data2.extend([test_u])
	data1.close();data2.close()		
	return zip(list_data1,list_data2)

def temporal_to_nwb(exptpath,meta,output_filename='temporal_dataset.nwb'):
	#requires sync .h5 in a subfolder called sync
	#         stimulus pkls for contrast natural movies, repeated natural movie, and binary noise to be in a subfolder called stim
	#         the kilosort .npy files in the base directory
	
    syncdata = Dataset(glob.glob(exptpath+'/sync/*.h5')[0])
    #get the info from the events file (ephys)
    events = ephys.oe.loadEvents(os.path.join(exptpath,'experiment1_all_channels_0.events'))
    start_time = int(open(os.path.join(exptpath,'experiment1_messages_0.events'),'r').readlines()[0].split(' ')[0])/meta['sampling_rate']  
    ch0 = events['timestamps'][events['channel']==0]
    ch1 = events['timestamps'][events['channel']==1]
    ch2 = events['timestamps'][events['channel']==2]
    ch3 = events['timestamps'][events['channel']==3]
    
    #get the info from the sync file; convert to seconds with @ sampling rate of 100kHz
    bit1_rising = syncdata.get_rising_edges(1).astype(np.float64) / 100000.; 
    bit2_rising = syncdata.get_rising_edges(2).astype(np.float64) / 100000.;
    bit3_rising = syncdata.get_rising_edges(3).astype(np.float64) / 100000.;

    #get the offset between the start of sync and the start of ephys
    sync_offset = ch0[2] / 30000. - start_time - bit2_rising[0]

    #calulate stimulus timings
    #spontaneous
    spontaneous_start = events['timestamps'][events['channel']==2][1] / float(events['header']['sampleRate']) - start_time
    spontaneous_end = events['timestamps'][events['channel']==2][2] / float(events['header']['sampleRate']) - start_time
    
    #flicker
    flicker_timestamps_s = bit3_rising[1:][:50] + sync_offset   
    
    #contrast movies
    movies_start_time = bit3_rising[np.where(bit3_rising > flicker_timestamps_s[-1])[0][0]] + sync_offset
    temp = bit1_rising[np.where(bit1_rising + sync_offset > movies_start_time)[0]] + sync_offset
    gaps = temp[1:] - temp[:-1]
    next_event_after_big_gap = np.where(gaps > 1.)[0]+3
    contrast_movies_starts = np.zeros(1+len(next_event_after_big_gap)/2);
    contrast_movies_starts[0]=movies_start_time
    contrast_movies_starts[1:]=bit1_rising[next_event_after_big_gap[1::2]] + sync_offset
    contrast_pkls = {'8': pkl.load(open(glob.glob(exptpath+'/stim/*_08-*.pkl')[0])),
                   '16': pkl.load(open(glob.glob(exptpath+'/stim/*_16-*.pkl')[0])),
                   '32': pkl.load(open(glob.glob(exptpath+'/stim/*_32-*.pkl')[0])),
                   '64': pkl.load(open(glob.glob(exptpath+'/stim/*_64-*.pkl')[0])),
                   '100': pkl.load(open(glob.glob(exptpath+'/stim/*_100-*.pkl')[0])),
                   }
    contrast_times = {}
    for i,contrast in enumerate(['100','32','8','64','16']):
        contrast_times[contrast] = bit1_rising[np.where((bit1_rising + sync_offset > contrast_movies_starts[i]) & (bit1_rising + sync_offset < contrast_movies_starts[i+1]))[0]] + sync_offset
        contrast_indices = [frame[0] for frame in contrast_pkls[contrast]['bgsweepframes']]
        contrast_times[contrast] = contrast_times[contrast][contrast_indices]

    #binary noise
    temp = bit2_rising[np.where(bit2_rising + sync_offset > movies_start_time)[0]] + sync_offset
    gaps = temp[1:] - temp[:-1]
    next_event_after_big_gap = np.where(gaps > 10.)[0][0] + 1
    start_of_binary = temp[next_event_after_big_gap]
    temp = bit2_rising[np.where(bit2_rising + sync_offset > start_of_binary)[0]] + sync_offset
    gaps = temp[1:] - temp[:-1]
    next_event_after_big_gap = np.where(gaps > 1.)[0][0] + 1
    start_of_movie = temp[next_event_after_big_gap]
    binary_times = bit1_rising[np.where((bit1_rising + sync_offset > start_of_binary) & (bit1_rising + sync_offset < start_of_movie))[0]] + sync_offset
    
    #movie repeated at high contrast
    movie_pkl = pkl.load(open(glob.glob(exptpath+'/stim/*_TOE2-*.pkl')[0]))
    movie_indices = [frame[0] for frame in movie_pkl['bgsweepframes']]
    movie_times = bit1_rising[np.where(bit1_rising + sync_offset > start_of_movie)[0]] + sync_offset
    movie_times = movie_times[movie_indices]
#=====================================================================================================================        


#=====================================================================================================================        
    #create file
    nwb_file = nwbtools.ExtracellularEphysFile(modify=True,meta=meta)
    #add ephys
    nwb_file.add_clustering(exptpath,name='LGN')
    
    # add stimulus stuff to nwb
    #spontaneous
    nwb_file.add_stimulus_information(np.array([spontaneous_start,spontaneous_end]),
                                      data=np.array([1]),
                                      start_time=0,
                                      name='spontaneous epoch 1',
                                      features=['none'],
                                      features_units=['two timestamps, the beginning and end'])
    
    #flicker
    stimulus_movie = pkl.load(open('/Volumes/danmac1_data_local_1/M186117/17.17.1_M186117_2016-06-03_07-26-20_500_500_EXT_insertion2to3.2mm_frozen/2016-06-03-073856511000.pkl'))['flicker']
    nwb_file.add_stimulus_information(flicker_timestamps_s,
                                      data=stimulus_movie,
                                      start_time=0,
                                      name='flicker',
                                      features=['luminance'],
                                      features_units=['luminance at 20Hz'])
    #movie contrast
    toe2 = np.load('/Volumes/Public/Dan/ephys/TOE2.npy')
    for i,contrast in enumerate(contrast_pkls.keys()):
        movie_pkl = contrast_pkls[contrast]
        stimulus_movie = toe2
        nwb_file.add_stimulus_information(contrast_times[contrast] ,
                                          data=np.array([1]),
                                          start_time=0,
                                          name='natural movie: TOE2 contrast: '+contrast,
                                          features=['time-space-space'],
                                          features_units=['luminance'])
    
    #binary
    stimulus_movie = pkl.load(open('/Volumes/Public/Dan/ephys/M227382/insertion2/stim/2016-08-08-120403155000.pkl')).T
    nwb_file.add_stimulus_information(binary_times,
                                      data=stimulus_movie,
                                      start_time=0,
                                      name='binary',
                                      features=['time-space-space'],
                                      features_units=['luminance'])
    
    #movie x100
    movie_pkl = pkl.load(open(glob.glob(exptpath+'/stim/*_TOE2-*.pkl')[0]))
    #stimulus_movie = toe2
    nwb_file.add_stimulus_information(movie_times,
                                      data=np.array([1]),
                                      start_time=0,
                                      name='natural movie: TOE2',
                                      features=['time-space-space'],
                                      features_units=['luminance'])
    
    #spontaneous
    spontaneous_start2 =movie_times[-1] #end of last stimulus
    nwb_file.add_stimulus_information(np.array([spontaneous_start2,spontaneous_start2+600.]),
                                      data=np.array([0,1]),
                                      start_time=0,
                                      name='spontaneous epoch 2',
                                      features=['none'],
                                      features_units=['two timestamps, the beginning and end'])

    
    nwb_file.close()
#=====================================================================================================================        

#=================================================================================================




#=================================================================================================
#------------operations on continuous traces-------------------------------------
#=================================================================================================

def get_chunk(mm,start,end,channels,sampling_rate=30000):
    chunk = mm[start*sampling_rate*len(channels):end*sampling_rate*(len(channels))]
    return np.reshape(chunk,(len(channels),-1),order='F')  * 0.195

#filter a bit of continuous data. uses butterworth filter.
def filterTrace(trace, low, high, sampleHz, order):
    low = float(low)
    high = float(high)
    nyq = 0.5 * sampleHz
    low = low / nyq
    high = high / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered = scipy.signal.lfilter(b, a, trace)
    return filtered
    
#developmental filter version. not used.
def filterTrace_hard(trace, low, high, sampleHz, order):
    low = float(low)
    high = float(high)
    nyq = 0.5 * sampleHz
    low = low / nyq
    high = high / nyq
    scipy.signal.band_stop_obj()
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered = scipy.signal.lfilter(b, a, trace)
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
#            print('skipped trial: '+str(i+1))
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

#returns the root mean squared of the input data    
def RMS(data,start=0,window=0,despike=False):
    start = start * samplingRate# sampling rate
    if window == 0:
        window = len(data)
    else:
        window = window * samplingRate # sampling rate
    #chunk = filterTrace(data[start:start+window], 70, 6000, 25000, 3)[200:window]
    chunk = data[start:start+window] - np.mean(data[start:start+window])
    if despike:
        chunk = despike_trace(chunk,threshold=180)
    return np.sqrt(sum(chunk*chunk)/float(len(chunk)))

def despike_trace(trace,threshold_sd = 2.5,**kwargs):
	if 'threshold' in kwargs.keys():
		threshold = kwargs['threshold']
	else:
		threshold = np.mean(trace)+threshold_sd*np.std(trace)
		
	spike_times_a = plt.mlab.cross_from_below(trace,threshold)
	spike_times_b = plt.mlab.cross_from_below(trace,-1*threshold)
	for spike_time in np.concatenate((spike_times_b,spike_times_a)):
		if spike_time > 30 and spike_time < len(trace)-30:
			trace[spike_time - 20:spike_time + 20] = 0#np.random.uniform(-1*threshold,threshold,60)
	return trace

def spikeamplitudes_trace(trace,threshold_sd = 3.0,**kwargs):
	if 'threshold' in kwargs.keys():
		threshold = kwargs['threshold']
	else:
		threshold = np.mean(trace)+threshold_sd*np.std(trace)
		
	spike_times_a = plt.mlab.cross_from_below(trace,threshold)
	amps=[]
	for spike_time in spike_times_a:
		if spike_time > 30 and spike_time < len(trace)-30:
			amps.extend([np.max(np.abs(trace[spike_time-30:spike_time+30]))])
	if not len(amps) > 0:
		amps= [0]
	return np.sort(amps)[len(amps)*0.9] / 5.0

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
def powerspectrum(data,start=0,window=0,plot=False,ymin=1e-24,ymax=1e8,title='',samplingRate=2500):
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
        plt.xlim(xmin=0.01);
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
        window = len(data)
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
    spectrum, frequencies = plt.mlab.cohere(x,y,Fs=float(samplingRate),NFFT=int(samplingRate)/5)
    if returnval:
        if type(returnval) is float:
            return np.interp(returnval,frequencies,spectrum)
        if type(returnval) is tuple:
            return np.trapz(spectrum[np.where(frequencies==returnval[0])[0]:np.where(frequencies==returnval[1])[0]],dx=5.0)  
    else:
        return (spectrum, frequencies)
#=================================================================================================


#=================================================================================================
#------------operations on spike trains-------------------------------------
#------------includes some plotting of spike trains-------------------------------------
#=================================================================================================
#compute and optionally plot a peri-stimulus time histogram
#plot is a line plot, with options for error display [bars or shaded]
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
        #print('start: ' + str(start)+'   end: ' + str(end))

    variance = np.std(bytrial,axis=0)/binsize/np.sqrt((len(triggers)))
    hist = np.mean(bytrial,axis=0)/binsize
    edges = np.linspace(-pre+binsize,post+binsize,numbins)

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
	
#find the x and y position of an average waveform, given the probe geometry
#plot is a bar plot
def psth(times,triggers,timeDomain=False,pre=0.5,post=1,binsize=0.05,ymax=75,output='fig',name='',color=[1,1,1],axes=None,labels=True,sparse=False,labelsize=18):
    peris=[]#np.zeros(len(triggers),len(times))
    if timeDomain:
        samplingRate = 1.0
    else:
        samplingRate = samplingRate
        
    times = np.array(times).astype(float) / samplingRate
    triggers = np.array(triggers).astype(float) / samplingRate
    
    for i,t in enumerate(triggers):
        peris.append(np.array(times).astype(float)-float(t))
    peris = np.array(peris)
    peris=peris.flatten()

    numbins = (post+pre) / binsize 
    (hist,edges) = np.histogram(peris,int(numbins),(-pre,post))
    hist /= float(len(triggers)*binsize)


    if output == 'fig':
        if axes == None:
            plt.figure()
            axes=plt.gca()
        f=axes.bar(edges[:-1],hist,width=binsize,color=color)
        axes.set_xlim(-pre,post)
        axes.set_ylim(0,ymax)
        if sparse:
            axes.set_xticklabels([])
            axes.set_yticklabels([])
        else:
            if labels:
                axes.set_xlabel(r'$time \/ [s]$',fontsize=20)
                axes.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=20)
                axes.tick_params(axis='both',labelsize=labelsize)
        axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
        axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')
        axes.set_title(name)
        return axes
    if output == 'hist':
        return (hist,edges)    
    if output == 'p':
        return peris

#compute and optionally show peri-stimulus time histograms for a list of arrays of times
def psthlist(timesdict,timeslist,onsets,pre=0.5,post=1,binsize=0.05,ymax=50,output='fig'):
    fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,np.shape(timeslist)[0],sharey=True) 
    axes=(ax1,ax2,ax3,ax4)
    for j,s in enumerate(timeslist):
        peris=[]#np.zeros(len(onsets),len(times))
        times = timesdict[s]
        times = np.array(times).astype(float) / samplingRate
        onsets = np.array(onsets).astype(float) / samplingRate
        
        for i,t in enumerate(onsets):
            peris.append(np.array(times).astype(float)-float(t))
        peris = np.array(peris) 
        peris=peris.flatten()
    
        numbins = (post+pre) / binsize 
        (hist,edges) = np.histogram(peris,int(numbins),(-pre,post))
        
        hist /= float(len(onsets)*binsize)
    
        print(j)
        axis = axes[j]
        print(axis)
        axis.hist(peris,int(numbins),(-pre,post),color=[1,1,1])
        axis.set_xlim(-pre,post)
        axis.set_ylim(0,ymax)
        if j==0:
            axis.set_xlabel(r'$time \/ [s]$',fontsize=14)
            axis.set_ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=14)
        axis.set_title(s)
    
    plt.show()    
    return fig

def raster(times,triggers,pre=0.5,timeDomain=False,post=1,yoffset=0,output='fig',name='',color='#00cc00',linewidth=0.5,axes=None,labels=True,sparse=False,labelsize=18,axis_labelsize=20,error='',alpha=0.5,ms=2,**kwargs):
	post = post + 1
	if timeDomain:
		samplingRate = 1.0
	else:
		samplingRate = samplingRate
        
	times = np.array(times).astype(float) / samplingRate + pre
	triggers = np.array(triggers).astype(float) / samplingRate
	bytrial = [];
	
	if axes == None and output!='data':
		plt.figure()
		axes=plt.gca()
		
	for i,t in enumerate(triggers):
		if len(np.where(times >= t - pre)[0]) > 0 and len(np.where(times >= t + post)[0]) > 0:
			start = np.where(times >= t - pre)[0][0]
			end = np.where(times >= t + post)[0][0]
			bytrial.append(np.array(times[start:end-1])-t)
			if output!='data':
		#		print(np.ones(len(np.array(times[start:end-1])-t))*i+1)
				axes.plot(np.array(times[start:end-1])-t,np.ones(len(np.array(times[start:end-1])-t))*i+1,"|",mew=0.5,ms=ms,color=color)
	if output!='data':
		#axes.set_xlim(-pre,post)
		axes.set_title(name)
	
		if sparse:
			cleanAxes(axes,total=True)
		else:
			if labels:
				axes.set_xlabel(r'$time \/ [s]$',fontsize=16)
				axes.set_ylabel(r'$trial \/ number$',fontsize=16)
				axes.tick_params(axis='both',labelsize=labelsize)
				axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
				axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')
	if output == 'fig':
		return (plt.gcf(),plt.gca())
	if output=='data':
		return bytrial


def raster_singletrial(cells,trigger,pre=0.5,timeDomain=False,post=1,yoffset=0,output='fig',name='',color='#00cc00',linewidth=0.5,axes=None,labels=True,sparse=False,labelsize=18,axis_labelsize=20,error='',alpha=0.5,**kwargs):
	post = post + 1
	if timeDomain:
		samplingRate = 1.0
	else:
		samplingRate = samplingRate
        
	t = float(trigger) / samplingRate
	bycell = [];
	
	if axes == None:
		plt.figure()
		axes=plt.gca()
		
	for i,cell in enumerate(cells.keys()):
		times = cells[cell]['times']
		if len(np.where(times >= t - pre)[0]) > 0 and len(np.where(times >= t + post)[0]) > 0:
			start = np.where(times >= t - pre)[0][0]
			end = np.where(times >= t + post)[0][0]
			bycell.append(np.array(times[start:end-1])-t)
			axes.plot(np.array(times[start:end-1])-t,np.ones(len(np.array(times[start:end-1])-t))*i+1,'|',linewidth=1,color=color50[i%50])
	axes.set_xlim(-pre,post)
	axes.set_title(name)
	
	if sparse:
		cleanAxes(axes,total=True)
	else:
		if labels:
			axes.set_xlabel(r'$time \/ [s]$',fontsize=16)
			axes.set_ylabel(r'$cell \/ number$',fontsize=16)
			axes.tick_params(axis='both',labelsize=labelsize)
			axes.spines['top'].set_visible(False);axes.yaxis.set_ticks_position('left')
			axes.spines['right'].set_visible(False);axes.xaxis.set_ticks_position('bottom')
	if output == 'fig':
		return (plt.gcf(),plt.gca())
	if output=='data':
		return bycell
        


#compute the tuning over a given parameter from a PSTH
def psth_tuning(data,unit,param,params,paramtimesdict,window=1.33,binsize=0.02,savepsth=False,path=r'C:\Users\danield\Desktop\data'):
    tun_y = np.zeros(len(params))
    tun_x = np.zeros(len(params))
    i=0
    for p in params:
        print(p)
        if savepsth:
            f=psth(data[unit]['times'],paramtimesdict[param+str(p)],pre=0,post=window,binsize=binsize)
            f.savefig(os.path.join(path,'unit'+unit+param+str(p)+'_psth.eps'),format='eps')     
        
        (hist,edges) = psth(data[unit]['times'],paramtimesdict[param+str(p)],pre=0,post=window,binsize=binsize,output='hist')        
        tun_y[i] = np.mean(hist)
        tun_x[i] = p
        i+=1
        
    plt.plot(tun_x,tun_y,'ko-');
    plt.xscale('log');
    plt.xlim(7,101);
    plt.ylabel(r'$firing \/ rate \/ [Hz]$',fontsize=14);
    plt.xlabel(r'$contrast \/ $[%]',fontsize=14);
    f=plt.gcf()
    return f

#compute the latency to first reseponse from a PSTH
def psth_latency(data,bins,pre=None,binsize=None, sd = 2.5,smooth=False,offset=0):
    if smooth:
        data = scipy.signal.savgol_filter(data,5,3)
    if pre is None:
        pre = bins[0]
    if binsize == None:
        binsize = bins[1]-bins[0]
    startbin = np.where(bins>0)[0][0]
    baseline = np.mean(data[:startbin])
    threshold = baseline + np.std(data[:startbin])*sd +0.2
    crossings = plt.mlab.cross_from_below(data[startbin:],threshold)
    if len(crossings)>0:
        crossing = crossings[0]#the first bin above the threshold
        chunk = np.linspace(data[crossing+startbin-1],data[crossing+startbin],100)
        bin_crossing = plt.mlab.cross_from_below(chunk,threshold)
        latency =(crossing-1)*(1000*binsize)+bin_crossing/100.0 * (1000*binsize) 
    else:
        #print('response did not exceed threshold: '+str(threshold)+', no latency returned')
        return None
    return latency[0] - offset

#compute the latency to first reseponse from a PSTH
def psth_area(data,bins,pre=None,binsize=None, sd = 3,time=0.2):
    if pre is None:
        pre = bins[0]
    if binsize == None:
        binsize = bins[1]-bins[0]
    startbin = np.where(bins>0)[0][0]
    baseline = np.mean(data[:startbin])
    threshold = baseline + np.std(data[:startbin])*sd +0.2
    crossings = plt.mlab.cross_from_below(data[startbin:],threshold)
    if len(crossings)>0:
        try:
            area = np.trapz(np.abs(data[startbin:startbin+np.ceil(time/binsize)]) - baseline)
            return area
        except: return None 
        print('response did not exceed threshold: '+str(threshold)+', no area returned')
        return None
    

#returns the F1 frequency of a response. requires the frequency to specified.
#computed in the Fourier domain.
def f1(inp,freq):
    #a = [];[a.append(np.cos(np.linspace(0,2*np.pi,np.shape(inp)[0]/numcycles))) for i in range(numcycles)]; cos = np.concatenate(a)
    #b = [];[b.append(np.sin(np.linspace(0,2*np.pi,np.shape(inp)[0]/numcycles))) for i in range(numcycles)]; sin = np.concatenate(b)           

    ps = np.fft.fft(inp)**2 / np.sqrt(len(inp))
    #ps = (ps*ps.conj()).real
    #plt.plot(ps)
    return np.abs(ps[freq])


#measure the cross-correlogram between two input spike trains.
#uses geometric mean for normalization
#returns x,y of the CCG.
def ccg(train1,train2,binrange,binsize):
    diffs = []
    count=0
    if len(train1) > 1 and len(train2) > 1:
        for spiketime_train1 in train1:
            #for spiketime_train2 in train2:
            if train2[-1] > spiketime_train1 + binrange[0]: # if there are any spikes from train2 after the start of the window 
                start = np.where(train2 > spiketime_train1 + binrange[0])[0][0]

                if train2[-1] > spiketime_train1 + binrange[1]:#set the end of train2 to only relevant window around this spike
                    end = np.where(train2 > spiketime_train1 + binrange[1])[0][0]
                else:
                    end = len(train2)

                for spiketime_train2 in train2[start:end]:
                    diffs.extend([float(spiketime_train1) - float(spiketime_train2)])
                    count+=1
        diffs = np.array(diffs)*-1
        hist,edges = np.histogram(diffs,bins=(binrange[1]-binrange[0])/binsize,range=binrange)
        return (hist / float(len(train1)))*100,edges
        #return (hist / float(len(train1)*len(train2) / 2.)  ,edges)
        #return ((hist / (len(train1) * len(train2)) / 2.)*100 * binsize,edges)
    else:
        print('input spiketrains not long enough: 1:'+str(len(train1))+' 2:'+str(len(train2)))
        return [0],[0,0]



#compute a spike-triggered average on three dimensional data. this is typically a movie of the stimulus
#TODO: should be modified to take data of any shape (for example, an LFP trace) to average.
def sta(spiketimes,data,datatimes,taus=(np.linspace(-10,280,30)),exclusion=None,samplingRateInkHz=25):
    output = {}
    for tau in taus:
        avg = np.zeros(np.shape(data[:,:,0]))
        count = 0
        for spiketime in spiketimes:
            if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.6:
                if exclusion is not None: #check to see if there is a period we are supposed to be ignoring, because of eye closing or otherwise
                    if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.6:
                         index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                         avg += data[:,:,index]
                else:
                	index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                	avg += data[:,:,index]
                count+=1
        output[str(int(tau))]=avg/count
    return output

#should compute a spike-triggered average on n-dimensional data.
#this is typically a movie of the stimulus of shape [frames,x,y], but could be any shape, such as a 1-d noise stimulus or continuous trace.
#as of now, limited to computation of the average along the first axis of 'data'. 
def sta2(spiketimes,data,datatimes,taus=(np.linspace(-10,280,30)),exclusion=None,samplingRateInkHz=30,time_domain=False):
    output = {}
    if time_domain:
        taus = taus / 1000.
    else:
        taus = (taus * samplingRateInkHz)/1000.
		
    for tau in taus:
        avg = np.zeros(np.shape(data[0]))
        count = 0
        for spiketime in spiketimes:
            if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                if exclusion is not None: #check to see if there is a period we are supposed to be ignoring, because of eye closing or otherwise
                    if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                         index = (np.where(datatimes > (spiketime - tau))[0][0]-1) % np.shape(data)[0]
                         avg += data[index]
                else:
                	index = (np.where(datatimes > (spiketime - tau))[0][0]-1) % np.shape(data)[0]
                	avg += data[index]
                count+=1
        output[str(tau)]=avg/count
    return output

#compute a spike-triggered average from sparse noise stimulus
def sta_sparse(spiketimes,data,datatimes,datashape,sparseshape=(120,120),taus=(np.linspace(-10,580,60)),exclusion=None,samplingRateInkHz=25,sign='both'):
    output = {}
    for tau in taus:
        avg = np.zeros((datashape[0],datashape[1]))
        count = 0
        for spiketime in spiketimes:
            addFrameToAvg = False
            if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                if exclusion is not None: #check to see if there is a period we are supposed to be ignoring, because of eye closing or otherwise
                    if  spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                        addFrameToAvg = True
                else:
                    addFrameToAvg = True
                    
                if addFrameToAvg:
                    stimID = np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]
                    if sign == 'both':
                        avg[data[stimID][2]/sparseshape[0]+datashape[0]/2][data[stimID][3]/sparseshape[1]+datashape[1]/2]+=data[stimID][0]
                    if sign=='dark'and data[stimID][0]==-1:
                        avg[data[stimID][2]/sparseshape[0]+datashape[0]/2][data[stimID][3]/sparseshape[1]+datashape[1]/2]+=-1#data[stimID][0]
                    if sign=='bright' and data[stimID][0]==1:
                        avg[data[stimID][2]/sparseshape[0]+datashape[0]/2][data[stimID][3]/sparseshape[1]+datashape[1]/2]+=1#data[stimID][0]
                    count+=1;
                    addFrameToAvg=False
        output[str(int(tau))]=avg/count
    return output

#returns a normalized ratio of the amplitude of to PSTHs
#psth1 - psth2
#_____________
#psth2 + psth2
def ratio_responses(psth1,psth2,peak=None,window=5):
	if peak is not None: # resample the input psths. if peak==None, don't resample
		if peak == 'max':
			if np.max(psth1) > np.max(psth2):
				peak = int(np.where(psth1 == np.max(psth1))[0][0])
			else:
				peak = int(np.where(psth2 == np.max(psth2))[0][0])
		if peak == 'rising':
			peak = None
			
		if type(peak) == int: 	
			pass
		else:
			print('error finding peak of response')
			
		psth1 = psth1[peak-window:peak+window]
		psth2 = psth2[peak-window:peak+window]
	
	return (np.trapz(psth1) - np.trapz(psth2)) / (np.trapz(psth1) + np.trapz(psth2))

def Welchs(a,b,equal_var=False,output=False):
    print('a: '+str(np.mean(a))+u' \u00B1 '+str(np.std(a)))
    print('b: '+str(np.mean(b))+u' \u00B1 '+str(np.std(b)))

    t_stat, p_stat = ttest_ind(a,b,equal_var=False)
    
    print('t: '+str(t_stat)+'  p: '+str(p_stat))
    if output:
        return (t_stat, p_stat)
#=================================================================================================



#=================================================================================================
#------------operations on stimulus sync-------------------------------------
#=================================================================================================

#parse an aibs SweepStim dictionary to get the times a grating was presented
def getgratingparamtimes(param,value,times,stimpkl):
    order = stimpkl['bgsweeporder']
    table = stimpkl['bgsweeptable']
    if len(order) >= len(times):
        index = ['contrast','z','tf','sf','x','y','ori'].index(param)
        onsets = []
        for i,time in enumerate(times):
            if table[order[i]][index] == value:
                onsets.append(time)
        if len(onsets) > 0:
            return onsets
        else:
            print(''.join((param,' = ',str(value),' not found.')))
    else:
        print('number of times did not match number of gratings. ')
		
		
#convenience function for checking the inter-stimulus times in a noise experiment
#compares times in the stimulus pickle and those derived from a sync signal, usually in the ephys timebase.
def checkTimes(ephys_times,pkl_times):
    ephystimes = np.array(ephys_times)/25000.0
    pkltimes = pkl_times+(a[1]-pkl_times[1])
    corrected_times = np.zeros(len(pkltimes))
    e=0;p=0
    for i in range(len(pkltimes)):
        if e < len(ephystimes):
            print( np.abs(ephystimes[e]-pkltimes[p]))
            if np.abs(ephystimes[e]-pkltimes[p]) < 1.0: 
                corrected_times[i] = ephystimes[e]
                e+=1
                p+=1
            else:
                corrected_times[i] = 0.1
                e+=0
                p+=1      
    return corrected_times
#=================================================================================================



#=================================================================================================
#---------plotting functions-------------------------------------------------
#=================================================================================================
def placeAxesOnGrid(fig,dim=[1,1],xspan=[0,1],yspan=[0,1],wspace=None,hspace=None):
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

#plotting function for showing a bit of data
def showChunk(data,start,window,channelmap=[],prefix = '100_CH',filt=False,yrange=400,zero=False,sparse=False):  
    if channelmap == []:
        channelmap = data.keys()
    if len(data.keys()) >= np.shape(channelmap)[0]*np.shape(channelmap)[1]:
        fig,axes = plt.subplots(nrows=np.shape(channelmap)[1],ncols=np.shape(channelmap)[0])
        for row in range(np.shape(channelmap)[1]):
            for column in range(np.shape(channelmap)[0]):
                if filt != False:
                    response = filterTrace(data[prefix+str(channelmap[column][row]).replace(prefix,'')]['data'],filt[0],filt[1],samplingRate,3)[start:start+window]
                else:
                    response = data[prefix+str(channelmap[column][row]).replace(prefix,'')]['data'][start:start+window]  
                if zero:
                	response = response - response[0]
                axes[row][column].plot(response)
                axes[row][column].set_ylim(-yrange,yrange)
                axes[row][column].text(10,100,'CH'+str(channelmap[column][row]))
# 				if sparse:
# 					if row > 0 or column > 0:
# 						axes[row][column].set_frame_on(False)
# 						axes[row][column].set_xticklabels('',visible=False)
# 						axes[row][column].set_xticks([])
# 						axes[row][column].set_yticklabels('',visible=False)
# 						axes[row][column].set_yticks([])
# 					#.set_title(prefix+str(channelmap[column][row]).replace(prefix,''))
# 					#axes[row][column].set_title(prefix+str(channelmap[column][row]).replace(prefix,''))
# 				else:
# 					axes[row][column].text(10,100,'CH'+str(channelmap[column][row]))
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.set_size_inches(11,18.5)
    else:
        print('channel map does not match input data')

#variant of plotting function for showing a bit of data
def showChunk2(data,start,window,channelmap=[],prefix = '100_CH',filt=False,yrange=40,zero=False,sparse=False,vertline=False,color='k'):
    if channelmap == []:
        channelmap = data.keys()
    if len(data.keys()) >= np.shape(channelmap)[0]*np.shape(channelmap)[1]:
        fig,axes = plt.subplots(1,ncols=np.shape(channelmap)[0])
        for row in range(np.shape(channelmap)[1]):
            for column in range(np.shape(channelmap)[0]):
            	if vertline is not False:
            		if row==0:
            			if type(vertline) is tuple:
            				for v in vertline:
            					axes[column].axvline(x=v,axes=axes[column],color='r')
            				axes[column].axvspan(vertline[0],vertline[1],axes=axes[column],facecolor='0.5',alpha=0.5)
            			if type(vertline) is int:
            				axes[column].axvline(x=vertline,axes=axes[column],color='r')
                # if filt != False:
                #     response = filterTrace(data[prefix+str(channelmap[column][row]).replace(prefix,'')]['data'],filt[0],filt[1],samplingRate,3)[start:start+window]
                # else:
                #     response = data[prefix+str(channelmap[column][row]).replace(prefix,'')]['data'][start:start+window]  
                # if zero:
                #     response = response - response[0]
                # axes[column].plot(response+row*yrange*-1,color=color)
                # #axes[column].set_ylim(-yrange,yrange)
                # if sparse:
                #     axes[column].set_frame_on(False)
                #     axes[column].set_xticklabels('',visible=False)
                #     axes[column].set_xticks([])
                #     axes[column].set_yticklabels('',visible=False)
                #     axes[column].set_yticks([])
				     
        fig.subplots_adjust(hspace=0)
        plt.tight_layout()
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        fig.set_size_inches(11,8)
    else:
        print('channel map does not match input data')        

#plotting function for showing all of the data in an array
def showData(data,channelmap=[],yrange=400,prefix = '100_CH',vertline=False, sparse=True,zero=False):
    if channelmap == []:
        channelmap = data.keys()
    if len(np.shape(channelmap)) == 1:
        rows = np.shape(channelmap)[0]
        cols = 1
        channelmap=[channelmap]
    else:
        rows = np.shape(channelmap)[1]
        cols = np.shape(channelmap)[0]
#    if len(data.keys()) == rows * cols:
    fig,axes = plt.subplots(nrows=rows,ncols=cols)
#        2D = np.zeros((np.shape(channelmap)[1],np.shape(channelmap)[0],len(data[data.keys()[0]]['data'])))
    for row in range(rows):
        for column in range(cols):
            response = data[prefix+str(channelmap[column][row]).replace(prefix,'')]['data']
            if zero:
            	response = response - response[0]
 #               2D[row,column,:]=response
            axes[row][column].plot(response,'-k')
            axes[row][column].set_ylim(-yrange,yrange)
            if vertline is not False:
                axes[row][column].axvline(x=vertline,axes=axes[row][column],color='r')
            if sparse:
                if row > 0 or column > 0:
                    axes[row][column].set_frame_on(False)
                    axes[row][column].set_xticklabels('',visible=False)
                    axes[row][column].set_xticks([])
                    axes[row][column].set_yticklabels('',visible=False)
                    axes[row][column].set_yticks([])
                    #.set_title(prefix+str(channelmap[column][row]).replace(prefix,''))
                    axes[row][column].set_title(prefix+str(channelmap[column][row]).replace(prefix,''))
            else:
                axes[row][column].text(10,100,'CH'+str(channelmap[column][row]))
    if sparse:
        fig.subplots_adjust(hspace=0)
        #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        #plt.setp([a.get_yticklabels() for a in fig.axes[:-1]], visible=False)
    fig = plt.gcf()
    fig.set_size_inches(11,38.5)
    return fig
#    else:
#        print('channel map does not match input data')

#smooth a 2D image, meant to be space-space of a receptive field
#size = number of pixels to smooth over
def smoothRF(img,size=3):
    smooth = ndimage.gaussian_filter(img,(size,size))
    return smooth

from scipy.signal import boxcar,convolve
def smooth_boxcar(data,boxcar_size):
    smoothed = convolve(data,boxcar(boxcar_size))/boxcar_size
    smoothed = smoothed[boxcar_size/2:len(data)+(boxcar_size/2)]
    return smoothed

#show the space-space plots of an already computed STRF for a range of taus.
def plotsta(sta,taus=(np.linspace(-10,280,30).astype(int)),colorrange=(-0.15,0.15),title='',taulabels=False,nrows=3,cmap=plt.cm.seismic,smooth=None):
    ncols = np.ceil(len(taus) / nrows ).astype(int)#+1
    fig,ax = plt.subplots(nrows,ncols,figsize=(10,6))
    titleset=False
    m=np.mean(sta[str(taus[3])])
    for i,tau in enumerate(taus):
        axis = ax[int(np.floor(i/ncols))][i%ncols]
        if smooth is None:
            img = sta[str(tau)].T 
        else:
            img = smoothRF(sta[str(tau)].T,smooth)

        axis.imshow(img,cmap=cmap,vmin=colorrange[0],vmax=colorrange[1],interpolation='none')
        axis.set_frame_on(False);
        axis.set_xticklabels('',visible=False);
        axis.set_xticks([]);
        axis.set_yticklabels('',visible=False);
        axis.set_yticks([])
        axis.set_aspect(1.0)
        if taulabels:
            axis.set_title('tau = '+str(tau),fontsize=8)
        if titleset is not True:
            axis.set_title(title,fontsize=12)
            titleset=True
        else:
            if tau == 0:
                axis.set_title('tau = '+str(tau),fontsize=12)
    plt.tight_layout()
    fig.show()


#function for fitting with a 2-dimensional gaussian.
def twoD_Gaussian(p, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x=p[0];y=p[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()
def fit_rf_2Dgauss(data,center_guess,width_guess=2,height_guess=2):
    dataToFit = data.ravel()
    x=np.linspace(0,np.shape(data)[0]-1,np.shape(data)[0])
    y=np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    x, y = np.meshgrid(x, y)
    popt,pcov = opt.curve_fit(twoD_Gaussian,(x,y),dataToFit,p0=(data[center_guess[1]][center_guess[0]], center_guess[1], center_guess[0], width_guess, height_guess, 0, 0))
    reshaped_to_space=(x,y,twoD_Gaussian((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,pcov,reshaped_to_space

def fit_rf_2Dgauss_centerFixed(data,center_guess,width_guess=2,height_guess=2):
    dataToFit = data.ravel()
    x=np.linspace(0,np.shape(data)[0]-1,np.shape(data)[0])
    y=np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    x, y = np.meshgrid(x, y)
    
    def twoD_Gaussian_fixed(p, amplitude,sigma_x, sigma_y, theta, offset):
        x=p[0];y=p[1]
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-center_guess[1])**2) + 2*b*(x-center_guess[1])*(y-center_guess[0]) 
                                + c*((y-center_guess[0])**2)))
        return g.ravel()
    popt,pcov = opt.curve_fit(twoD_Gaussian_fixed,(x,y),dataToFit,p0=(data[center_guess[0]][center_guess[1]], width_guess, height_guess, 0, 0))
    reshaped_to_space=(x,y,twoD_Gaussian_fixed((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,pcov,reshaped_to_space

def impulse(sta,center,taus = np.arange(-10,580,10).astype(int)):
	impulse = [sta[str(tau)][center[0]][center[1]] for tau in taus]
	return (taus,impulse)


#try to fit an already computed STRF by finding the maximum in space-time, fitting with a 2D gaussian in space-space, and pulling out the temporal kernel at the maximum space-space pixel.
#is very, very finicky right now, requires lots of manual tweaking. 
def fitRF(RF,threshold=None,fit_type='gaussian_2D',verbose=False,rfsizeguess=1.2,flipSpace=False,backup_center=None,zoom_int=10,zoom_order=5,centerfixed=False):
#takes a dictinary containing:
#   
# returns a dictionary containing:
#   the averaged spatial RF, 
#   the centroid of the fit [max fit]
#   a 2D gaussian fit of that spatial RF
#   the impulse response at the center of the fit
#   TODO: a fit of that impulse response with: ?? currently not defined.
    if np.isnan(RF[RF.keys()[0]][0][0]):#check to make sure there is any data in the STRF to try to fit. if not, return the correct data structure filled with None
        fit={};fit['avg_space_fit']=None;fit['params'] = None;fit['cov']=None ;fit['amplitude']=None ;fit['x']=None ;fit['y']=None ;fit['s_x']=None ;fit['s_y']=None ;fit['theta']=None ;fit['offset']=None;fit['center']=None;fit['peakTau']=None;fit['impulse']=None;fit['roughquality']=None
        return fit
    else:
        if 'fit' in RF.keys():
            trash = RF.pop('fit') # get rid of the dictionary entry 'fit'; we only want sta tau frames in this dictionary
        taus = [int(i) for i in RF.keys()]
        taus.sort()
        fit={}
        
        #========================================================================
        #find the taus to average over for the spatial RF
        #first, define the threshold; above this means there is a non-noise pixel somwhere in the space-space
        if threshold == None:
            #set to z sd above mean
            blank = (RF['-10']+RF['0']+RF['10'])/3. # define a blank, noise-only image
            threshold = np.mean(blank)+np.std(blank)*3.
            if verbose:
                print('threshold: '+str(threshold))
            
        #find the average space-space over only the range of non-noise good 
        avgRF = np.zeros(np.shape(RF[str(int(taus[0]))]))#initialize the average to blank.
        goodTaus = [40,50,60,70,80,90,100]#
        for tau in goodTaus:
            avgRF += RF[str(int(tau))]
        avgRF = avgRF / float(len(goodTaus))
        fit['avg_space']=avgRF
        #========================================================================   
        
        #====fit==================================================================
        maximum_deviation = 0;best_center = (0,0)
        for i in np.linspace(24,63,40):
            for j in np.linspace(10,49,40):
                imp_temp = impulse(RF,(i,j))
                if np.max(np.abs(imp_temp[1])) > maximum_deviation:
                    best_center = (i,j)
                    maximum_deviation = np.max(np.abs(imp_temp[1]))
        center = best_center
        imp_temp = impulse(RF,center)
        if verbose:
            print('peak frame tau: '+str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]])))
            print('peak center   : '+str(center))
            print('peak value    : '+str(RF[str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]]))][center[0],center[1]]))
        peak_frame = RF[str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]]))]
        peak = peak_frame[center[0],center[1]]
        #center = (np.where(np.abs(smoothRF(peak_frame,1)) == np.max(np.abs(smoothRF(peak_frame,1))))[0][0],np.where(np.abs(smoothRF(peak_frame,1)) == np.max(np.abs(smoothRF(peak_frame,1))))[1][0])
        
        if verbose:
            print('peak amp: '+str(peak)+'  threshold: '+str(threshold))
        if np.abs(peak) > threshold * 1.0:
            peak_frame = smoothRF(zoom(peak_frame,zoom_int,order=zoom_order),0)
            fit['roughquality']='good'
        else:
            center = backup_center
            imp_temp = impulse(RF,center)
            peak_frame = RF[str(int(100))]
            peak = peak_frame[center[0],center[1]]
            peak_frame = smoothRF(zoom(peak_frame,zoom_int,order=zoom_order),0)
            print('could not find a peak in the RF, using center: '+str(center))
            print('peak amplitude: '+str(peak)+', threshold: '+str(threshold))
            fit['roughquality']='bad'
        fit['center']=center
        fit['center_guess']=center
        fit['fit_image']=peak_frame
        if verbose:
            print('center guess: '+str(center))
        
        #initialize some empty parameters
        fitsuccess=False;retry_fit = False
        best_fit = 10000000#initialize impossibly high
        fit['avg_space_fit']=None;fit['params']=None
        best_fit_output = ((None,None,None,None,None,None,None),600,None)
        try:
            if centerfixed:
                popt,pcov,space_fit = fit_rf_2Dgauss_centerFixed(peak_frame,(center[0]*zoom_int,center[1]*zoom_int),width_guess=rfsizeguess*zoom_int,height_guess=rfsizeguess*zoom_int)
            else:
                popt,pcov,space_fit = fit_rf_2Dgauss(peak_frame,(center[0]*zoom_int,center[1]*zoom_int),width_guess=rfsizeguess*zoom_int,height_guess=rfsizeguess*zoom_int)
        except:
            popt,pcov,space_fit=((None,0,0,0,0,0,0),600,np.zeros((64,64)))
        
        fit['avg_space_fit']=np.array(space_fit)/float(zoom_int)
        fit['params'] = popt    
        fit['cov']=pcov
        fit['amplitude']=popt[0]
        if centerfixed:
            fit['x']=center[1]
            fit['y']=center[0]
            fit['s_x']=popt[1] / float(zoom_int)
            fit['s_y']=popt[2] / float(zoom_int)
            fit['theta']=popt[3]
            fit['offset']=popt[4] / float(zoom_int)
        else:
            fit['x']=popt[1] / float(zoom_int)
            fit['y']=popt[2] / float(zoom_int)
            fit['s_x']=popt[3] / float(zoom_int)
            fit['s_y']=popt[4] / float(zoom_int)
            fit['theta']=popt[5]
            fit['offset']=popt[6] / float(zoom_int)
        #======================================================================== 
    
    #        
        #============get impulse======================================================================== 
        if verbose:
            print('center: '+str(center[0])+' '+str(center[1]))
            
        if fit['avg_space_fit'] is not None:
            center_h = (np.ceil(fit['y']),np.ceil(fit['x']))
            center_r = (np.round(fit['y']),np.round(fit['x']))
            center_l = (np.floor(fit['y']),np.floor(fit['x']))
        try:
            impuls_h = impulse(RF,center_h,taus)[1]
            impuls_r = impulse(RF,center_r,taus)[1]
            impuls_l = impulse(RF,center_l,taus)[1]
            if np.max(np.abs(impuls_h)) > np.max(np.abs(impuls_r)):
                if np.max(np.abs(impuls_h)) > np.max(np.abs(impuls_l)):
                    impuls = impuls_h
                    center = center_h
                else:
                    impuls = impuls_l
                    center= center_l
            else:
                if np.max(np.abs(impuls_r)) > np.max(np.abs(impuls_l)):
                    impuls= impuls_r
                    center= center_r
                else:
                    impuls= impuls_l
                    center= center_l
        except:
            impuls = np.zeros(len(taus))
        
        if fit_type == 'gaussian_2D':
            #get impulse at the 'center'
            if verbose:
                print('center from fit: '+str(center[0])+' '+str(center[1]))
            #impuls = [RF[str(tau)][center[0]][center[1]] for tau in taus]
            fit['impulse']=(np.array(taus),np.array(impuls))
            peakTau = taus[np.abs(np.array(impuls)).argmax()]
            peakTau = 80
            fit['peakTau']=peakTau
        
        fit['center_usedforfit'] = fit['center']
        fit['center_usedforimp'] = center
        fit['impulse']=(np.array(taus),np.array(impuls))
        #======================================================================== 
        
        return fit 


#convenience plotting method for showing spatial and temporal filters pulled from fitting an already computed STRF
def show_sta_fit(fit,colorrange=(0.35,0.65),cmap=plt.cm.seismic,title='',contour_levels=3):
    
    if fit is not None:
        fig = plt.figure(figsize=(6,2.75))
        ax_full_space = placeAxesOnGrid(fig,dim=(1,1),xspan=(0,0.32),yspan=(0,0.5))
        ax_zoom_space = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.33,0.63),yspan=(0,0.5))
        ax_zoom_space_filtered = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.56,0.86),yspan=(0,0.5))
        ax_impulse = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.1,0.7),yspan=(0.54,1.0))
        
        ax_impulse.plot(fit['impulse'][0],np.zeros(len(fit['impulse'][0]))+0.5,'r-')
        ax_impulse.plot(fit['impulse'][0],fit['impulse'][1]);
        ax_impulse.set_ylim(colorrange[0],colorrange[1]) 
        ax_impulse.set_xlabel('time (msec)') 
        ax_impulse.set_ylabel('spike probability')
        #ax_impulse.text(1,1,str(np.mean(fit['cov'])))
        
        ax_full_space.plot([fit['y']],[fit['x']],'k+',markersize=6.0)#([fit['center'][0]],[fit['center'][1]],'k+',markersize=6.0)
        ax_full_space.imshow(fit['avg_space'].T,
           cmap=cmap,
           vmin=colorrange[0],vmax=colorrange[1],
            interpolation='none');
        plt.setp(ax_full_space.get_xticklabels(),visible=False);ax_zoom_space_filtered.xaxis.set_major_locator(plt.NullLocator())
        plt.setp(ax_full_space.get_yticklabels(),visible=False);ax_zoom_space_filtered.yaxis.set_major_locator(plt.NullLocator())
        ax_full_space.set_title(title)
        ax_full_space.set_ylim(20,40)
        ax_full_space.set_xlim(12,52)
        ax_full_space.axvline(x=32,linewidth=1,color='r',ls='--')
        
        zoom_size = 5 #number of pixels in each direction from peak center to keep in zooms
        ax_zoom_space.imshow(fit['avg_space'],
                   cmap=cmap,
                   vmin=colorrange[0],vmax=colorrange[1],
                    interpolation='none');
        #ax_zoom_space.plot([fit['x']],[fit['y']],'k+',markersize=6.0)
        ax_zoom_space.set_xlim(fit['center'][1]-zoom_size,fit['center'][1]+zoom_size)
        ax_zoom_space.set_ylim(fit['center'][0]-zoom_size,fit['center'][0]+zoom_size) 
        plt.setp(ax_zoom_space.get_xticklabels(),visible=False);ax_zoom_space_filtered.xaxis.set_major_locator(plt.NullLocator())
        plt.setp(ax_zoom_space.get_yticklabels(),visible=False);ax_zoom_space_filtered.yaxis.set_major_locator(plt.NullLocator())
        

        ax_zoom_space_filtered.imshow(smoothRF(fit['avg_space'],1),
                   cmap=cmap,
                   vmin=colorrange[0]/3.,vmax=colorrange[1]/3.,
                    interpolation='none');
        
        #ax_zoom_space_filtered.plot([fit['x']],[fit['y']],'k+',markersize=6.0)
        ax_zoom_space_filtered.set_xlim(fit['center'][1]-zoom_size,fit['center'][1]+zoom_size)
        ax_zoom_space_filtered.set_ylim(fit['center'][0]-zoom_size,fit['center'][0]+zoom_size)
        plt.setp(ax_zoom_space_filtered.get_xticklabels(),visible=False);ax_zoom_space_filtered.xaxis.set_major_locator(plt.NullLocator())
        plt.setp(ax_zoom_space_filtered.get_yticklabels(),visible=False);ax_zoom_space_filtered.yaxis.set_major_locator(plt.NullLocator())
        
    
        

        
        if 'avg_space_fit' in fit.keys():
            if fit['avg_space_fit'] is not None:
                if np.max(np.abs(fit['impulse'][1])) > np.std(fit['impulse'][1])*2.:
                    ax_full_space.add_patch(Rectangle((fit['y']-zoom_size,fit['x']-zoom_size),zoom_size*2,zoom_size*2,fill=None,ls='dotted'))
                    ax_zoom_space.contour(fit['avg_space_fit'][0],
                                fit['avg_space_fit'][1],
                                fit['avg_space_fit'][2],
                                contour_levels)
                    ax_zoom_space_filtered.contour(fit['avg_space_fit'][0],
                                fit['avg_space_fit'][1],
                                fit['avg_space_fit'][2],
                                contour_levels)
                    ax_full_space.contour(fit['avg_space_fit'][1],
                                fit['avg_space_fit'][0],
                                fit['avg_space_fit'][2],
                                contour_levels)
                    ax_full_space.set_aspect('equal')   

		#                   vmin=-0.08,vmax=0.08);
        #plt.tight_layout()
        return fig

#another method for displaying the average waveform of the spikes from a specific unit 
def plotWaveform(ax,data,yrange=200,offset=100,geometry=(np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2]),sampling_rate=25000.0,centroid=None,window=3,color='#FFFFFF'):
    for column,c in enumerate(geometry):
        for row,channel in enumerate(c):
            x = np.linspace(column*(len(data[:,0])+1),column*(len(data[:,0])+1)+len(data[:,0]),len(data[:,0]))
            x = x / sampling_rate
            #x = np.linspace(column*(len(data[:,0])+1)/sampling_rate,column*(len(data[:,0])+1)/sampling_rate+len(data[:,0])/sampling_rate,len(data[:,0]))
            #x = np.linspace(column*1.1,column*1.1+len(data[:,0])/sampling_rate,len(data[:,0]))
            y = data[:,channel-1]- data[:,channel-1][0] - offset*row
            if centroid is not None:
                if np.abs(row - centroid) < window:
                    ax.plot(x,y,color=color,linewidth=10)
                else:
                    ax.plot(x,y,color='#BEC6C4',alpha=0.5)
            else:
                ax.plot(x,y,'-k')

def get_waveform_duration(waveform, sampling_rate=30000.):
    w = resample(waveform,200)#upsample to smooth the data
    time = np.linspace(0,len(waveform)/sampling_rate,200)
    peak = np.where(w==np.max(w))[0][0]
    trough = np.where(w==np.min(w))[0][0]
    half_max = w[peak]/2.
#     if w[peak] > np.abs(w[trough]):
# 		dur = time[plt.mlab.cross_from_below(w,half_max)[0]]-time[plt.mlab.cross_from_above(w,half_max)[0]]
#     else:
# 	    dur = time[plt.mlab.cross_from_above(w,half_max)[0]]-time[plt.mlab.cross_from_below(w,half_max)[0]]
    if w[peak] > np.abs(w[trough]):
        dur =   time[peak:][np.where(w[peak:]==np.min(w[peak:]))[0][0]] - time[peak] 
    else:
        dur =   time[trough:][np.where(w[trough:]==np.max(w[trough:]))[0][0]] - time[trough] 
    return dur

def get_waveform_PTratio(waveform, sampling_rate=30000.):
    w = resample(waveform,200)#upsample to smooth the data
    time = np.linspace(0,len(waveform)/sampling_rate,200)
    peak = np.where(w==np.max(w))[0][0]
    trough = np.where(w==np.min(w))[0][0]
    ratio = w[peak]/abs(w[trough])
    if ratio > 1.:
        return 1.
    else:
        return w[peak]/abs(w[trough])

def get_waveform_repolarizationslope(waveform, sampling_rate=30000.,window=30):
    w = resample(waveform,200)#upsample to smooth the data
    time = np.linspace(0,len(waveform)/sampling_rate,200)
    trough = np.where(w==np.min(w))[0][0]
    return linregress(time[trough:trough+window],w[trough:trough+window])[0]


def plotExchangeFromPSTH(data,unit,setcolor,UV='#4B0082',Green='#6B8E23',**kwargs):
    if 'axis' in kwargs.keys():
        axis = kwargs['axis']
    else:
        plt.figure();axis=plt.gca()
    
    if 'color_exchange_'+setcolor in data['all']['lgn'][unit].keys():
        if data['all']['lgn'][unit]['color_exchange_'+setcolor][0][0] is not None:
            test_contrasts_o=np.array([-1.0,-0.8,-0.6,-0.5,-0.4,-0.2,0.0])
            uv_green_ratio_opp = np.array([(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][17:26])+np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][63:72])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:9])*2.0) for test_contrast in test_contrasts_o])
            test_contrasts_s=np.array([1.0,0.8,0.6,0.5,0.4,0.2,0.0])
            uv_green_ratio_same = np.array([(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][17:26])+np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][63:72])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:9])*2.0) for test_contrast in test_contrasts_s])

            if setcolor == 'uv':
                axis.plot(test_contrasts_o/0.4,uv_green_ratio_opp/np.max([uv_green_ratio_opp,uv_green_ratio_same]),'--o',color=Green)
                axis.plot(test_contrasts_s/0.4,uv_green_ratio_same/np.max([uv_green_ratio_opp,uv_green_ratio_same]),'-o',color=Green)
            else:
                axis.plot(test_contrasts_o/0.4,uv_green_ratio_opp/np.max([uv_green_ratio_opp,uv_green_ratio_same]),'--o',color=UV)
                axis.plot(test_contrasts_s/0.4,uv_green_ratio_same/np.max([uv_green_ratio_opp,uv_green_ratio_same]),'-o',color=UV)

            axis.set_ylim(-1.1,1.1);axis.set_xlim(-3,3)
            axis.set_ylabel(r'$normalized response$',fontsize=12)
            axis.tick_params(axis='both',labelsize=10)
            axis.spines['top'].set_visible(False);axis.yaxis.set_ticks_position('left')
            axis.spines['right'].set_visible(False);axis.xaxis.set_ticks_position('bottom')   
            axis.set_xlabel(r'$set:test \/ Ratio$',fontsize=12)
            axis.axhline(y=0,color='k',linestyle='--')
            #axis.set_title(expt_names[i]+' unit: '+unit)
            
def plotExchangeFromPSTH_2(data,unit,setcolor,UV='#4B0082',Green='#6B8E23',dataonly=False,**kwargs):
    if 'axis' in kwargs.keys():
        axis = kwargs['axis']
    else:
        if dataonly == False:
            plt.figure();axis=plt.gca()
    
    if 'color_exchange_'+setcolor in data['all']['lgn'][unit].keys():
        if data['all']['lgn'][unit]['color_exchange_'+setcolor][0][0] is not None:
            test_contrasts_o=np.array([-1.0,-0.8,-0.6,-0.5,-0.4,-0.2,0.0])
            set_test_ratio_opp = np.array([np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][10:60])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][-60:]) for test_contrast in test_contrasts_o])
            #set_test_ratio_opp = np.array([((np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][13:63])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][63:78]))/(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:10])*3.5+0.1))-1 for test_contrast in test_contrasts_o])
            #set_test_ratio_opp = np.array([((np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][13:63]))/(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:10])*5.0+0.1))-1 for test_contrast in test_contrasts_o])
            #set_test_ratio_opp=np.array([(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][17:26])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][63:72])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:9])) for test_contrast in test_contrasts_o])
            
            test_contrasts_s=np.array([1.0,0.8,0.6,0.5,0.4,0.2,0.0])
            set_test_ratio_same = np.array([np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][10:60])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][-60:]) for test_contrast in test_contrasts_s])
            #set_test_ratio_same = np.array([((np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][13:63])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][63:78]))/(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:10])*3.5+0.1))-1 for test_contrast in test_contrasts_s])
            #set_test_ratio_same = np.array([((np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][13:63]))/(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:10])*5.0+0.1))-1 for test_contrast in test_contrasts_s])
            #set_test_ratio_same=np.array([(np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][17:26])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][63:72])-np.trapz(data['all']['lgn'][unit]['color_exchange_'+setcolor][test_contrast][0][:9])) for test_contrast in test_contrasts_s])

            if dataonly:
                return (set_test_ratio_opp/np.max(np.abs(zip(set_test_ratio_opp,set_test_ratio_same))),
                set_test_ratio_same/np.max(np.abs(zip(set_test_ratio_opp,set_test_ratio_same))))
            else:
                if setcolor == 'green':
                    x=-0.4/(test_contrasts_o);x[-1]=test_contrasts_s[0]/0.4+0.2
                    axis.plot(x,set_test_ratio_opp/np.max(np.abs(zip(set_test_ratio_opp,set_test_ratio_same))),'--o',color=UV)
                    #axis.plot(x,set_test_ratio_opp,'--o',color=Green)
                    x=0.4/(test_contrasts_s);x[-1]=test_contrasts_s[0]/0.4+0.2
                    axis.plot(x,set_test_ratio_same/np.max(np.abs(zip(set_test_ratio_opp,set_test_ratio_same))),'-o',fillstyle='none',color=UV)
                    #axis.plot(x,set_test_ratio_same,'-o',fillstyle='none',color=Green)
                else:
                    axis.plot(test_contrasts_o/-0.4,set_test_ratio_opp/np.max(np.abs(zip(set_test_ratio_opp,set_test_ratio_same))),'--o',color=Green)
                    #axis.plot(test_contrasts_o/-0.4,set_test_ratio_opp,'--o',color=UV)
                    axis.plot(test_contrasts_s/0.4,set_test_ratio_same/np.max(np.abs(zip(set_test_ratio_opp,set_test_ratio_same))),'-o',fillstyle='none',color=Green)
                    #axis.plot(test_contrasts_s/0.4,set_test_ratio_same,'-o',fillstyle='none',color=UV)
    
                axis.set_ylim(-1.1,1.1);axis.set_xlim(0,2.7)
                axis.set_ylabel(r'$normalized response$',fontsize=12)
                axis.tick_params(axis='both',labelsize=10)
                axis.spines['top'].set_visible(False);axis.yaxis.set_ticks_position('left')
                axis.spines['right'].set_visible(False);axis.xaxis.set_ticks_position('bottom')   
                axis.set_xlabel(r'$green:UV \/ Ratio$',fontsize=12)
                axis.axhline(y=0,color='k',linestyle='--')
                #axis.set_title(expt_names[i]+' unit: '+unit)
            

            
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

def summarize_color(data,unit,color_folder=''):
    fig = plt.figure(figsize=(8.5,11))

    #isi, time
    ax_isi =  placeAxesOnGrid(fig,dim=(1,1),xspan=(0,0.15),yspan=(0,0.08))#plt.subplot2grid((gridsize_x,gridsize_y),(0,0),colspan=2)
    ax_time =  placeAxesOnGrid(fig,dim=(1,1),yspan=(0,0.08),xspan=(0.25,1.0))#plt.subplot2grid((gridsize_x,gridsize_y),(0,3),colspan=8)
    if 'isi' not in data['all']['lgn'][unit].keys():
        if len(data['all']['lgn'][unit]['times'])>1000:
            spks = np.array(data['all']['lgn'][unit]['times'][:1000])/25000.0
        else:
            spks = np.array(data['all']['lgn'][unit]['times'])/25000.0
        data['all']['lgn'][unit]['isi']=ephys.ccg(spks,spks,(-100,100),0.5)
        data['all']['lgn'][unit]['isi'][1][np.where(np.logical_and(data['all']['lgn'][unit]['isi'][0]>-1.0,data['all']['lgn'][unit]['isi'][0]<1.0))]=0
    ax_isi.plot(data['all']['lgn'][unit]['isi'][1][1:],data['all']['lgn'][unit]['isi'][0],'k');cleanAxes(ax_isi,bottomLabels=True)
    dump = ephys.psth_line(np.array(data['all']['lgn'][unit]['times'])/25000.0,
                             [0],
                             pre=0,post=data['all']['lgn'][unit]['times'][-1]/25000.0 - 1,
                             binsize=10,
                             error='shaded',timeDomain=True,sparse=False,
                             labelsize=8,axis_labelsize=10,
                             axes=ax_time,color='k')
    ax_time.set_ylim(0,np.max(plt.gca().get_lines()[0].get_ydata()))
    box_props = dict(boxstyle='round',fc='w',alpha=0.6)
    for i,start_time in enumerate(data['info']['start_times']):
        nm = ''
        for clause in data['info']['folder_order'][i].split('_'):
            if clause[-1].isdigit():
                pass
            else:
                nm = nm+'_'+clause
        ax_time.axvline(start_time/25000.,linestyle='dashed',color='r')
        ax_time.text(start_time/25000.,np.max(plt.gca().get_lines()[0].get_ydata())/1.4,nm[1:],size=8,rotation=90,ha='left',va='bottom',bbox=box_props)
    ax_isi.set_title(data['name']+' '+str(unit))

    #flash
    ymax = 0
    for i,key in enumerate(['flash_uv','flash_green']):
        if key in data['all']['lgn'][unit].keys():
            for k in data['all']['lgn'][unit][key]:
                if np.max(data['all']['lgn'][unit][key][k][0]) > ymax:
                    ymax = np.ceil(np.max(data['all']['lgn'][unit][key][k][0]))
    ax_flashes=placeAxesOnGrid(fig,dim=(3,3),xspan=(0,0.30),yspan=(0.13,0.4))
    ax_bright_uv = ax_flashes[0][1]
    ax_bright_uv.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#000000'))
    ax_bright_uv.plot(data['all']['lgn'][unit]['flash_uv']['bright'][1],data['all']['lgn'][unit]['flash_uv']['bright'][0],color=UV);ax_bright_uv.set_xlim(-0.05,0.3);ax_bright_uv.set_ylim(0,ymax);
    ax_dark_uv = ax_flashes[1][1]#plt.subplot2grid((gridsize_x,gridsize_y),(flashrowstart+2,1),rowspan=2)
    ax_dark_uv.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#ffffff',alpha=0.2))
    ax_dark_uv.plot(data['all']['lgn'][unit]['flash_uv']['dark'][1],data['all']['lgn'][unit]['flash_uv']['dark'][0],color=UV);ax_dark_uv.set_xlim(-0.05,0.3);ax_dark_uv.set_ylim(0,ymax);
    ax_bright_green = ax_flashes[0][0]#plt.subplot2grid((gridsize_x,gridsize_y),(flashrowstart,0),rowspan=2)
    ax_bright_green.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#000000'))
    ax_bright_green.plot(data['all']['lgn'][unit]['flash_green']['bright'][1],data['all']['lgn'][unit]['flash_green']['bright'][0],color=Green);ax_bright_green.set_xlim(-0.05,0.3);ax_bright_green.set_ylim(0,ymax);
    ax_bright_green.set_xticks([0,0.2]);ax_bright_green.set_yticks([0,ymax])
    ax_dark_green = ax_flashes[1][0]#plt.subplot2grid((gridsize_x,gridsize_y),(flashrowstart+2,0),rowspan=2)
    ax_dark_green.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#ffffff',alpha=0.2))
    ax_dark_green.plot(data['all']['lgn'][unit]['flash_green']['dark'][1],data['all']['lgn'][unit]['flash_green']['dark'][0],color=Green);ax_dark_green.set_xlim(-0.05,0.3);ax_dark_green.set_ylim(0,ymax);                             
    ax_dark_green.set_yticks([0,ymax])
    ax_uv = ax_flashes[2][1]#plt.subplot2grid((gridsize_x,gridsize_y),(flashrowstart+4,1),rowspan=2)
    ax_uv.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#ffffff',linestyle='dotted'))
    ax_uv.plot(data['all']['lgn'][unit]['flash_uv']['bright'][1],data['all']['lgn'][unit]['flash_uv']['bright'][0],color=UV);ax_uv.set_xlim(-0.05,0.3);ax_uv.set_ylim(0,ymax);
    ax_uv.plot(data['all']['lgn'][unit]['flash_uv']['dark'][1],data['all']['lgn'][unit]['flash_uv']['dark'][0],color=UV);ax_uv.set_xlim(-0.05,0.3);ax_uv.set_ylim(0,ymax);
    ax_uv.set_xticks([0,0.2]);
    ax_green = ax_flashes[2][0]#plt.subplot2grid((gridsize_x,gridsize_y),(flashrowstart+4,0),rowspan=2)
    ax_green.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#ffffff',linestyle='dotted'))
    ax_green.plot(data['all']['lgn'][unit]['flash_green']['bright'][1],data['all']['lgn'][unit]['flash_green']['bright'][0],color=Green);ax_green.set_xlim(-0.05,0.3);ax_green.set_ylim(0,ymax);
    ax_green.plot(data['all']['lgn'][unit]['flash_green']['dark'][1],data['all']['lgn'][unit]['flash_green']['dark'][0],color=Green);ax_green.set_xlim(-0.05,0.3);ax_green.set_ylim(0,ymax);                             
    ax_green.set_xticks([0,0.2]);ax_green.set_yticks([0,ymax])
    ax_bright =  ax_flashes[0][2]#plt.subplot2grid((gridsize_x,gridsize_y),(flashrowstart,2),rowspan=2)
    ax_bright.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#000000'))
    ax_bright.plot(data['all']['lgn'][unit]['flash_uv']['bright'][1],data['all']['lgn'][unit]['flash_uv']['bright'][0],color=UV);ax_bright.set_xlim(-0.05,0.3);ax_bright.set_ylim(0,ymax);
    ax_bright.plot(data['all']['lgn'][unit]['flash_green']['bright'][1],data['all']['lgn'][unit]['flash_green']['bright'][0],color=Green);ax_bright.set_xlim(-0.05,0.3);ax_bright.set_ylim(0,ymax);
    ax_dark = ax_flashes[1][2]#plt.subplot2grid((gridsize_x,gridsize_y),(flashrowstart+2,2),rowspan=2)
    ax_dark.add_patch(patches.Rectangle((0,0),0.05,ymax,facecolor='#ffffff',alpha=0.2))
    ax_dark.plot(data['all']['lgn'][unit]['flash_uv']['dark'][1],data['all']['lgn'][unit]['flash_uv']['dark'][0],color=UV)#ax_dark.set_xlim(-0.05,0.5);ax_dark.set_xlim(0,100);
    ax_dark.plot(data['all']['lgn'][unit]['flash_green']['dark'][1],data['all']['lgn'][unit]['flash_green']['dark'][0],color=Green);ax_dark.set_xlim(-0.05,0.3);ax_dark.set_ylim(0,ymax);                             
    ax_dark.set_xticks([0,0.2]);
    cleanAxes(ax_bright_uv);cleanAxes(ax_dark_uv)   
    cleanAxes(ax_bright_green,leftLabels=True);cleanAxes(ax_dark_green,leftLabels=True)
    cleanAxes(ax_green,leftLabels=True,bottomLabels=True);cleanAxes(ax_uv,bottomLabels=True)  
    cleanAxes(ax_bright);cleanAxes(ax_dark,bottomLabels=True) 
    ax_flashes[2][2].set_visible(False)

    #exchange
    ymax = 0
    for i,key in enumerate(['color_exchange_uv','color_exchange_green']):
        if key in data['all']['lgn'][unit].keys():
            for k in data['all']['lgn'][unit][key]:
                if np.max(data['all']['lgn'][unit][key][k][0]) > ymax:
                    ymax = np.ceil(np.max(data['all']['lgn'][unit][key][k][0]))
    test_contrasts = [-1.0,-0.8,-0.4,0.0,0.4,0.8,1.0]
    ax_exhange_ratios =placeAxesOnGrid(fig,dim=(1,1),xspan=(0.58,0.84),yspan=(0.14,0.38))  #plt.subplot2grid((gridsize_x,gridsize_y),(exchangerowstart+2,6),rowspan=3,colspan=3)
    if 'color_exchange_green' in data['all']['lgn'][unit].keys():
        plotExchangeFromPSTH_2(data,unit,'green',axis=ax_exhange_ratios)
        ax_exchange_green = placeAxesOnGrid(fig,dim=(len(test_contrasts),1),xspan=(0.90,1.0),yspan=(0.13,0.4))
        for i,test_contrast in enumerate(test_contrasts):
            ax = ax_exchange_green[i] #plt.subplot2grid((gridsize_x,gridsize_y),(i+exchangerowstart,4))
            ax.add_patch(patches.Rectangle((0,ymax/2.),1,test_contrast*(ymax/2.),facecolor=UV,alpha=0.2))
            ax.add_patch(patches.Rectangle((0,ymax/2.),1,0.4*(ymax/2.),facecolor=Green,alpha=0.2))
            ax.axhline(ymax/2.,linestyle='dotted')
            ax.plot(data['all']['lgn'][unit]['color_exchange_green'][test_contrast][1],data['all']['lgn'][unit]['color_exchange_green'][test_contrast][0],color=UV)
            ax.set_xlim(-0.1,1.35)
            ax.set_ylim(0,ymax)
            ax.set_xticks([0,1.]);ax.set_yticks([0,ymax])
            if i < len(test_contrasts)-1:
                cleanAxes(ax)
            else:
                cleanAxes(ax,bottomLabels=True,leftLabels=True)
    if 'color_exchange_uv' in data['all']['lgn'][unit].keys():        
        ax_exchange_uv = placeAxesOnGrid(fig,dim=(len(test_contrasts),1),xspan=(0.36,0.46),yspan=(0.13,0.4))
        plotExchangeFromPSTH_2(data,unit,'uv',axis=ax_exhange_ratios)
        for i,test_contrast in enumerate(test_contrasts):
            ax = ax_exchange_uv[i]#plt.subplot2grid((gridsize_x,gridsize_y),(i+exchangerowstart,9))
            ax.add_patch(patches.Rectangle((0,ymax/2.),1,test_contrast*(ymax/2.),facecolor=Green,alpha=0.2))
            ax.add_patch(patches.Rectangle((0,ymax/2.),1,0.4*(ymax/2.),facecolor=UV,alpha=0.2))
            ax.axhline(ymax/2.,linestyle='dotted')
            ax.plot(data['all']['lgn'][unit]['color_exchange_uv'][test_contrast][1],data['all']['lgn'][unit]['color_exchange_uv'][test_contrast][0],color=Green)
            ax.set_xlim(-0.1,1.35)
            ax.set_ylim(0,ymax)
            ax.set_xticks([0,1.]);ax.set_yticks([0,ymax])
            if i < len(test_contrasts)-1:
                cleanAxes(ax)
            else:
                cleanAxes(ax,bottomLabels=True,leftLabels=True)

    #STAs
    if 'sta_uv' in data['all']['lgn'][unit].keys() and 'sta_green' in data['all']['lgn'][unit].keys():
        taus = [10,30,60,90,120,150,180,210,240,270]
        ax_stas=placeAxesOnGrid(fig,dim=(2,len(taus)),xspan=(0,0.75),yspan=(0.46,0.6))

        
        data['all']['lgn'][unit]['sta_uv_fit'] = ephys.fitRF(data['all']['lgn'][unit]['sta_uv'],goodTaus=[80]) 
        data['all']['lgn'][unit]['sta_green_fit'] = ephys.fitRF(data['all']['lgn'][unit]['sta_green'],goodTaus=[80]) 
        if  np.abs(data['all']['lgn'][unit]['sta_uv_fit']['avg_space'][data['all']['lgn'][unit]['sta_uv_fit']['center']]) < np.abs(data['all']['lgn'][unit]['sta_green_fit']['avg_space'][data['all']['lgn'][unit]['sta_green_fit']['center']]):
            data['all']['lgn'][unit]['sta_uv_fit'] = ephys.fitRF(data['all']['lgn'][unit]['sta_uv'],center=data['all']['lgn'][unit]['sta_green_fit']['center'],goodTaus=data['all']['lgn'][unit]['sta_green_fit']['goodTaus']) 
        else:
            data['all']['lgn'][unit]['sta_green_fit'] = ephys.fitRF(data['all']['lgn'][unit]['sta_green'],center=data['all']['lgn'][unit]['sta_uv_fit']['center'],goodTaus=data['all']['lgn'][unit]['sta_uv_fit']['goodTaus']) 
        
        
        #ax_fitUV.imshow(data['all']['lgn'][unit]['sta_uv_fit']['avg_space'],cmap=plt.cm.seismic,clim=(-0.3,0.3))
        #ax_fitUV.set_xlim(data['all']['lgn'][unit]['sta_uv_fit']['center'][0]-9,data['all']['lgn'][unit]['sta_uv_fit']['center'][0]+9)
        #ax_fitUV.set_ylim(data['all']['lgn'][unit]['sta_uv_fit']['center'][1]-9,data['all']['lgn'][unit]['sta_uv_fit']['center'][1]+9)
        #cleanAxes(ax_fitUV,total=True)
        for i,tau in enumerate(taus):
            ax = ax_stas[0][i]#plt.subplot2grid((gridsize_x,gridsize_y),(stasrowstart,i))
            ax.imshow(data['all']['lgn'][unit]['sta_uv'][str(tau)],cmap=plt.cm.seismic,clim=(-0.2,0.2))
            #ax.set_xlim(data['all']['lgn'][unit]['sta_uv_fit']['center'][1]-9,data['all']['lgn'][unit]['sta_uv_fit']['center'][1]+9)
            #ax.set_ylim(data['all']['lgn'][unit]['sta_uv_fit']['center'][0]-9,data['all']['lgn'][unit]['sta_uv_fit']['center'][0]+9)
            #ax.set_xlim(5,35);ax.set_ylim(25,55)
            ax.set_title(str(tau-16));cleanAxes(ax,total=True)

        ax_impGreen  = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.78,1),yspan=(0.45,0.60))#plt.subplot2grid((gridsize_x,gridsize_y),(stasrowstart+1,10),colspan=2)
        ax_impGreen.plot(ephys.impulse(data['all']['lgn'][unit]['sta_green'],data['all']['lgn'][unit]['sta_green_fit']['center'])[0],ephys.impulse(data['all']['lgn'][unit]['sta_green'],data['all']['lgn'][unit]['sta_green_fit']['center'])[1],color=Green)#data['all']['lgn'][unit]['sta_green_fit']['impulse'][0],data['all']['lgn'][unit]['sta_green_fit']['impulse'][1],color=Green)
        ax_impGreen.plot(ephys.impulse(data['all']['lgn'][unit]['sta_uv'],data['all']['lgn'][unit]['sta_uv_fit']['center'])[0],ephys.impulse(data['all']['lgn'][unit]['sta_uv'],data['all']['lgn'][unit]['sta_uv_fit']['center'])[1],color=UV)#data['all']['lgn'][unit]['sta_green_fit']['impulse'][0],data['all']['lgn'][unit]['sta_green_fit']['impulse'][1],color=Green)
        #ax_impGreen.plot(data['all']['lgn'][unit]['sta_uv_fit']['impulse'][0],data['all']['lgn'][unit]['sta_uv_fit']['impulse'][1],color=UV)
        ax_impGreen.yaxis.tick_right();ax_impGreen.yaxis.set_label_position('right')
        ax_impGreen.set_xlim(-50,300);ax_impGreen.set_ylim(-0.3,0.3)
        ax_impGreen.set_xticks([0,100,200,300])
        ax_impGreen.annotate(str(data['all']['lgn'][unit]['sta_green_fit']['center']),xy=(0,-0.25),color=Green)
        ax_impGreen.annotate(str(data['all']['lgn'][unit]['sta_uv_fit']['center']),xy=(0,0.25),color=UV)

        cleanAxes(ax_impGreen,bottomLabels=True)
        for i,tau in enumerate(taus):
            ax = ax_stas[1][i]#plt.subplot2grid((gridsize_x,gridsize_y),(stasrowstart+1,i))
            ax.imshow(data['all']['lgn'][unit]['sta_green'][str(tau)],cmap=plt.cm.seismic,clim=(-0.2,0.2))
            #ax.set_xlim(data['all']['lgn'][unit]['sta_green_fit']['center'][1]-9,data['all']['lgn'][unit]['sta_green_fit']['center'][1]+9)
            #ax.set_ylim(data['all']['lgn'][unit]['sta_green_fit']['center'][0]-9,data['all']['lgn'][unit]['sta_green_fit']['center'][0]+9)
            #ax.set_xlim(5,35);ax.set_ylim(25,55)
            cleanAxes(ax,total=True)

    #contrast
    if 'contrast_green' in data['all']['lgn'][unit].keys() and 'contrast_uv' in data['all']['lgn'][unit].keys():
        contrasts = [0,0.04,0.08,0.16,0.24,0.32,0.48,0.64,1.0]
        ax_contrasts = placeAxesOnGrid(fig,dim=(2,len(contrasts)),xspan=(0,0.75),yspan=(0.65,0.81))
        highest = 0
        for i,contrast in enumerate(contrasts):
            if np.max(data['all']['lgn'][unit]['contrast_uv'][contrast][0][:150]) > highest:
                highest = np.max(data['all']['lgn'][unit]['contrast_uv'][contrast][0])
            if np.max(data['all']['lgn'][unit]['contrast_green'][contrast][0][:150]) > highest:
                highest = np.max(data['all']['lgn'][unit]['contrast_green'][contrast][0])
        f1s_uv=[]
        for i,contrast in enumerate(contrasts):
            ax = ax_contrasts[0][i]#plt.subplot2grid((gridsize_x,gridsize_y),(startrowcontrasts,i),rowspan=2)
            ax.plot(data['all']['lgn'][unit]['contrast_uv'][contrast][1][:150],data['all']['lgn'][unit]['contrast_uv'][contrast][0][:150],color=UV)
            ax.set_title(str(contrast))
            ax.set_ylim(0,highest)
            cleanAxes(ax)
            f1s_uv.append(ephys.f1(data['all']['lgn'][unit]['contrast_uv'][contrast][0][:150],4.5))

        f1s_green=[];
        for i,contrast in enumerate(contrasts):
            ax = ax_contrasts[1][i]#plt.subplot2grid((gridsize_x,gridsize_y),(startrowcontrasts+3,i),rowspan=2)
            ax.plot(data['all']['lgn'][unit]['contrast_green'][contrast][1][:150],data['all']['lgn'][unit]['contrast_green'][contrast][0][:150],color=Green)
            ax.set_ylim(0,highest)
            if i == 0:
                ax.set_xticks([0,1]);ax.set_yticks([0,highest])
                cleanAxes(ax,bottomLabels=True,leftLabels=True)
            else:
                cleanAxes(ax)
            f1s_green.append(ephys.f1(data['all']['lgn'][unit]['contrast_green'][contrast][0][:150],4.5))    
        ax_crf_green = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.78,1),yspan=(0.65,0.78))#plt.subplot2grid((gridsize_x,gridsize_y),(startrowcontrasts+3,len(contrasts)),rowspan=2,colspan=3)
        ax_crf_green.yaxis.tick_right();ax_crf_green.yaxis.set_label_position('right')
        ax_crf_green.plot(contrasts,f1s_green,'-o',color=Green)
        ax_crf_green.plot(contrasts,f1s_uv,'-o',color=UV)
        ax_crf_green.set_xlabel(r'$contrast \/ $[%]',fontsize=10);
        ax_crf_green.set_ylabel(r'$f1 \/ $[Hz]',fontsize=10);
        ax_crf_green.set_ylim(0,np.max([f1s_uv,f1s_green])+5);cleanAxes(ax_crf_green,bottomLabels=True,rightLabels=True)
        #ax_crf_uv.set_ylim(0,np.max([f1s_uv,f1s_green])+5);cleanAxes(ax_crf_uv,rightLabels=True)
        #ax_crf_uv.set_xscale('log');ax_crf_uv.set_xlim(np.min(contrasts)-0.02,np.max(contrasts))
        ax_crf_green.set_xscale('log');ax_crf_green.set_xlim(np.min(contrasts)-0.02,np.max(contrasts))

    #gratings
    if 'gratings_sf_isoluminant' in data['all']['lgn'][unit].keys() and 'gratings_sf_luminance' in data['all']['lgn'][unit].keys():
        sfs = np.sort(data['all']['lgn'][unit]['gratings_sf_luminance'].keys()).tolist()
        ax_sfs = placeAxesOnGrid(fig,dim=(2,len(sfs)),xspan=(0,0.75),yspan=(0.88,1.0))
        highest = 0
        for i,sf in enumerate(sfs):
            if np.max(data['all']['lgn'][unit]['gratings_sf_luminance'][sf][0][:150]) > highest:
                highest = np.max(data['all']['lgn'][unit]['gratings_sf_luminance'][sf][0])
            if np.max(data['all']['lgn'][unit]['gratings_sf_isoluminant'][sf][0][:150]) > highest:
                highest = np.max(data['all']['lgn'][unit]['gratings_sf_isoluminant'][sf][0])
        f1s_uv=[]
        for i,sf in enumerate(sfs):
            ax = ax_sfs[0][i]#plt.subplot2grid((gridsize_x,gridsize_y),(startrowgratings,i),rowspan=2)
            ax.plot(data['all']['lgn'][unit]['gratings_sf_isoluminant'][sf][1][:150],data['all']['lgn'][unit]['gratings_sf_isoluminant'][sf][0][:150],color='#3399ff')
            ax.set_title(str(sf))
            ax.set_ylim(0,highest)
            cleanAxes(ax)
            f1s_uv.append(ephys.f1(data['all']['lgn'][unit]['gratings_sf_isoluminant'][sf][0][:150],4.5))


        f1s_green=[];
        for i,sf in enumerate(sfs):
            ax = ax_sfs[1][i]#plt.subplot2grid((gridsize_x,gridsize_y),(startrowgratings+3,i),rowspan=2)
            ax.plot(data['all']['lgn'][unit]['gratings_sf_luminance'][sf][1][:150],data['all']['lgn'][unit]['gratings_sf_luminance'][sf][0][:150],color='k')
            ax.set_ylim(0,highest)
            if i == 0:
                ax.set_xticks([0,1]);ax.set_yticks([0,highest])
                cleanAxes(ax,bottomLabels=True,leftLabels=True)
            else:
                cleanAxes(ax)
            f1s_green.append(ephys.f1(data['all']['lgn'][unit]['gratings_sf_luminance'][sf][0][:150],4.5))    
        ax_sf_lum = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.78,1),yspan=(0.87,1.0))#plt.subplot2grid((gridsize_x,gridsize_y),(startrowgratings+3,len(sfs)),rowspan=2,colspan=3)
        ax_sf_lum.yaxis.tick_right();ax_sf_lum.yaxis.set_label_position('right')
        ax_sf_lum.plot(sfs,f1s_green,'-o',color='k')
        ax_sf_lum.plot(sfs,f1s_uv,'-o',color='#3399ff')
        ax_sf_lum.set_xlabel(r'$spatial freq. \/ $[cyc/o]',fontsize=10);
        ax_sf_lum.set_ylabel(r'$f1 \/ $[Hz]',fontsize=10);
        ax_sf_lum.set_ylim(0,np.max([f1s_uv,f1s_green])+5);cleanAxes(ax_sf_lum,bottomLabels=True,rightLabels=True)
        #ax_sf_iso.set_ylim(0,np.max([f1s_uv,f1s_green])+5);cleanAxes(ax_sf_iso,rightLabels=True)
        ax_sf_lum.set_xscale('log');ax_sf_lum.set_xlim(np.min(sfs)-0.01,np.max(sfs))
        #ax_sf_iso.set_xscale('log');ax_sf_iso.set_xlim(np.min(sfs)-0.01,np.max(sfs))

    #plt.tight_layout()
    fig.savefig(os.path.join(r'/Users/danieljdenman/Academics/manuscripts/2016_UV/plots/v2/eps',color_folder,data['name']+'_'+unit+'.eps'),format='eps')
    fig.savefig(os.path.join(r'/Users/danieljdenman/Academics/manuscripts/2016_UV/plots/v2/png',color_folder,data['name']+'_'+unit+'.png'),format='png')


def scatter_withcirclesize(ax,x,y,s,alpha=1.0,c='k',cmap=plt.cm.PRGn,**kwargs):
    if c != 'k':
        if type(c)==str:
            c = [c for dump in range(len(s))]
            cmap=None
        if type(c)==list:
            if len(c) == len(s):
                c = c
            else:
                print('incorrect number of colors specified.');return None
    else:
        c = ['k' for dump in range(len(s))]
    
    points=[]    
    for (x_i,y_i,r_i,c_i) in zip(x,y,s,c):
        points.append(Circle((x_i,y_i),radius=r_i))
    if cmap is not None:
        p = PatchCollection(points,cmap=cmap,alpha=alpha,clim=(-1,1))
        p.set_array(np.array(c))
        ax.add_collection(p)
    else:
        p = PatchCollection(points,color=c,alpha=alpha)
        ax.add_collection(p)
    #plt.colorbar(p)
    
def weibull(x,a,b):
    return 1 - np.exp(-(x/a))
def hyperbolicratio(C,r0,rmax,c50,n):
    return r0 + rmax * ( C**n / (c50**n+C**n))

def fit_hyperbolicratio(xdata,ydata,r0_guess,rmax_guess,c50_guess,n_guess):
    popt,pcov = opt.curve_fit(hyperbolicratio,xdata,ydata,p0=(r0_guess,rmax_guess,c50_guess,n_guess))
    r0,rmax,c50,n = popt
    return  r0,rmax,c50,n,pcov



#=================================================================================================






















# 
# 
# 
# #below is a class for plotting data in probe space. under development, and may be trashed.
# #-----------------------------------------------------------------------------------    
# #-----------------------------------------------------------------------------------    
# #-----------------------------------------------------------------------------------
# #-----------------------------------------------------------------------------------    
# class probeplot:
#     def __init__(self, parent=None, **kwargs):
#     	if 'axes' in kwargs.keys():
#     		self.fig = kwargs['fig']
#     		self.axes = kwargs['axes']
#     	else:
# 			self.fig = plt.figure(figsize = (6,12))
# 			self.axes = plt.gca()
# 			pg.mkQApp()
# 			#self.view = gl.GLViewWidget()
# 			#self.view.setBackgroundColor('k')
# 			self.waveformorder = np.array((np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2]))
# 		
#     def heatmap(self,data,geometry,cmin = 0, cmax = 0.25,prefix='101',offset=0):
#         #set up the figure
#         nrows = max(len(p) for p in geometry)
#         ncols = np.shape(geometry)[0]
#         fig = plt.figure(figsize = (2,12)) # set the figure size to be square
#         gs = gridspec.GridSpec(nrows, ncols,wspace=0.4, hspace=0.02, left = 0.2, right = 0.3, bottom = 0.05, top = 0.95) 
#         
#         #add the data
#         for column,c in enumerate(geometry):
#             for row,channel in enumerate(c):
#                #plot
#                 ax=plt.subplot(gs[nrows*column+row])
#                 im = ax.imshow([[data[prefix+'_CH'+str(channel+offset)]],[data[prefix+'_CH'+str(channel+offset)]]],vmin=cmin,vmax=cmax,cmap='gist_heat')
#                #cleanup [turn frame, ticks off]   
#                 ax.set_ylim(0,0.5);ax.set_xlim(0,0.5);            
#                 ax.set_frame_on(False);
#                 ax.set_xticklabels('',visible=False);
#                 ax.set_xticks([]);
#                 ax.set_yticklabels('',visible=False);
#                 ax.set_yticks([])         
#                 #ax.set_title(str(channel))
#        
#        #add a colorbar
#         cbar_ax = fig.add_axes([0.45, 0.15, 0.05, 0.7])
#         fig.colorbar(im,cax=cbar_ax)
#          
#          
#     def correlation(self,dic1,dic2,pnt='ko',**kwargs):
#         #if dic1.keys() == dic2.keys():
#         d1 = np.zeros(len(dic1.keys()))
#         d2 = np.zeros(len(dic1.keys()))
#         for i,key in enumerate(dic1.keys()):
#             if key in dic2.keys():
#                 d1[i] = dic1[key]
#                 d2[i] = dic2[key]
#             else:
#                 print(key)
#         plt.figure()
#         plt.plot(d1,d2,pnt)
#         if 'xmin' in kwargs.keys():
#             plt.xlim(xmin=kwargs['xmin']);
#         if 'ymin' in kwargs.keys():
#             plt.ylim(ymin=kwargs['ymin'])
#         if 'xmax' in kwargs.keys():
#             plt.xlim(xmax=kwargs['xmax']);
#         if 'ymax' in kwargs.keys():
#             plt.ylim(ymax=kwargs['ymax'])
#        # else:
#         #    print('input does not match')
#             
#         
#     def chunk(self,data,geometry,start=2,window=1,yrange=250,sHz = 25000,prefix='100',correct_offset=False):
#         #set up the figure
#         nrows = max(len(p) for p in geometry)
#         ncols = np.shape(geometry)[0]
#         #fig = plt.figure(figsize = (6,12)) # set the figure size to be square
#         gs = gridspec.GridSpec(nrows, ncols,wspace=0.1, hspace=0.1, left = 0.05, right = 0.95, bottom = 0.05, top = 0.95) 
#         
#         #add the data
#         for column,c in enumerate(geometry):
#             for row,channel in enumerate(c):
#                #plot
#                 #print('row: '+str(row)+'   column: '+str(column)+'     gs: '+str(row*ncols+column%ncols))
#                 ax=plt.subplot(gs[row*ncols+column%ncols])
#                 if correct_offset:
#                     im = ax.plot(data[prefix+'_CH'+str(channel)]['data'][start*sHz:start*sHz+window*sHz]-data[prefix+'_CH'+str(channel)]['data'][start*sHz-1],'k-')
#                 else:
#                     im = ax.plot(data[prefix+'_CH'+str(channel)]['data'][start*sHz:start*sHz+window*sHz],'k-')
#                 ax.set_ylim(-yrange,yrange);                
#                 ax.set_frame_on(False);
#                 ax.set_xticklabels('',visible=False);
#                 ax.set_xticks([]);
#                 ax.set_yticklabels('',visible=False);
#                 ax.set_yticks([])         
#                 ax.text(0,20,str(channel),size=10)
#         
#     def dataArray(self,data,geometry,yrange=250):
#         if np.shape(data)[1]==np.shape(geometry)[1]*np.shape(geometry)[0]:
#             #set up the figure
#             nrows = max(len(p) for p in geometry)
#             ncols = np.shape(geometry)[0]
#             fig = plt.figure(figsize = (6,12)) # set the figure size to be square
#             gs = gridspec.GridSpec(nrows, ncols,wspace=0.1, hspace=0.1, left = 0.05, right = 0.95, bottom = 0.05, top = 0.95) 
#             
#             #add the data
#             for column,c in enumerate(geometry):
#                 for row,channel in enumerate(c):
#                    #plot
#                     ax=plt.subplot(gs[nrows*column+row])
#                     im = ax.plot(data[:,channel-1])
#                     ax.set_ylim(-yrange,yrange);                
#                     ax.set_frame_on(False);
#                     ax.set_xticklabels('',visibleimport=False);
#                     ax.set_xticks([]);
#                     ax.set_yticklabels('',visible=False);
#                     ax.set_yticks([])         
#                     #ax.text(0,20,str(channel),size=10)
#         else:
#             print('size mismatch')
#             print(np.shape(data)[1])
#             print(np.shape(data)[1]*np.shape(data)[0])
#             
#     #inputs: data = spike dictionary, 
#     #        optional:     
#     #               yrange = 250 uV
#     #               offset = 20 uV
#     def waveformarray(self,data,yrange=250,offset=20,title='',geometry=(np.linspace(1,128,128)[::2][::-1],np.linspace(1,128,128)[1::2][::-1])):
#         if np.shape(data)[1]== np.sum([np.shape(geometry[column])[0] for column in range(np.shape(geometry)[0])]):
#             #set up the figure
#             nrows = 1#max(len(p) for p in geometry)
#             ncols = np.shape(geometry)[0]
#             fig,axes = plt.subplots(nrows=nrows, ncols=ncols)# = plt.figure(figsize = (6,12)) # set the figure size to be square
#             #gs = gridspec.GridSpec(nrows, ncols,wspace=0.1, hspace=0.1, left = 0.05, right = 0.95, bottom = 0.05, top = 0.95) 
# 
#             #add the data
#             for column,c in enumerate(geometry):
#                 for row,channel in enumerate(c):
#                    #plot
#                     #ax=plt.subplot(gs[nrows*column])
#                     axes[column].plot(data[:,channel-1]+offset*row,'-k')
#                     axes[column].set_ylim(-yrange,yrange);                
#                     axes[column].set_frame_on(False);
#                     axes[column].set_xticklabels('',visible=False);
#                     axes[column].set_xticks([]);
#                     axes[column].set_yticklabels('',visible=False);
#                     axes[column].set_yticks([])         
#                     #ax.text(0,20,str(channel),size=10)
#             fig.set_size_inches(4,10.5)
#             axes[0].set_title(title)
#         else:
#             print('size mismatch')
#             print(np.shape(data)[1])
#             print(np.shape(data)[1]*np.shape(data)[0])
#             
#     def waveformsurface(self,data,yrange=250):
#         geometry = self.waveformorder
#         #set up the figure
#         nrows = max(len(p) for p in geometry)
#         ncols = np.shape(geometry)[0]
#         ntimepoints = np.shape(data)[0]
# 
#         #surface = np.zeros(nrows * ntimepoints, ncols)
#         X = np.zeros(nrows * ntimepoints * ncols);
#         Y = np.zeros(nrows * ntimepoints * ncols);
#         Z = np.zeros(nrows * ntimepoints * ncols);
#         index = 0
#         
#         for column,c in enumerate(geometry):
#             #c = c[::-1]
#             for row,channel in enumerate(c):
#                 waveform = data[:,channel-1]
#                 for timepoint in range(len(waveform)):
#                     X[index] = column * ntimepoints + timepoint
#                     Y[index] = row 
#                     Z[index] = float(waveform[timepoint])/float(yrange) * -1
#                     index += 1
#          
#         triang = mtri.Triangulation(X,Y)
#         
#         #self.fig = plt.figure()
#         ax = self.fig.add_subplot(1, 1, 1, projection='3d')
#         ax.plot_trisurf(triang, Z, cmap=plt.cm.gist_heat)
#         ax.set_zlim(-0.3,1)
#         self.fig.set_size_inches(10,10)
#         #ax.initial_azim(-60)
#         #ax.initial_elev(-60)
#         mng = plt.get_current_fig_manager()
#         mng.window.showMaximized()
#         plt.show()
#         ax.set_zlim(-0.3,1)
# 
#     def waveformcloud(self,data,title='',geometry=(np.linspace(1,128,128)[::2][::-1],np.linspace(1,128,128)[1::2][::-1]),cmin=0,cmax=500):
#        #set up the figure
#         nrows = max(len(p) for p in geometry)
#         ncols = np.shape(geometry)[0]
#         fig = plt.figure(figsize = (2,12)) # set the figure size to be square
#         gs = gridspec.GridSpec(nrows, ncols,wspace=0.4, hspace=0.02, left = 0.2, right = 0.3, bottom = 0.05, top = 0.95) 
#         
#         #add the data
#         for column,c in enumerate(geometry):
#             for row,channel in enumerate(c):
#                #plot
#                 ax=plt.subplot(gs[nrows*column+row])
#                 value = np.trapz(np.abs(data[:,channel-1]))
#                 im = ax.imshow(np.array(([value,value],[value,value])),vmin=cmin,vmax=cmax,cmap='gist_heat')
#                #cleanup [turn frame, ticks off]   
#                 ax.set_ylim(0,0.5);ax.set_xlim(0,0.5);            
#                 ax.set_frame_on(False);
#                 ax.set_xticklabels('',visible=False);
#                 ax.set_xticks([]);
#                 ax.set_yticklabels('',visible=False);
#                 ax.set_yticks([])         
#                 #ax.set_title(str(channel))
#        
#        #add a colorbar
#         cbar_ax = fig.add_axes([0.45, 0.15, 0.05, 0.7])
#         fig.colorbar(im,cax=cbar_ax) 
#        
# 
#     def getwaveformspacetime(self,data,geometry,yrange=120):
# 		nrows = max(len(p) for p in geometry)
# 		ncols = np.shape(geometry)[0]# * 2
# 		ntimepoints = np.shape(data)[0]
# 		matrix = np.zeros((nrows,ncols, ntimepoints))   
# 		for column,c in enumerate(geometry):
# 			c = c[::-1]
# 			for row,channel in enumerate(c):
# 				waveform = data[:,channel-1]
# 				for timepoint in range(len(waveform)):
# 					matrix[row,column,timepoint] = float(waveform[timepoint])/float(yrange)
# 		return matrix[:,:,:]      
# 
#     def set_axes(self,ax):
# 		self.ax = self.axes[ax]        
# 		         
#     def waveform_animateblob(self,data,yrange=250,cmap=plt.cm.RdGy,geometry=np.array((np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2])),aspect = 0.2,ax=None):
#         #set up the figure
#         nrows = max(len(p) for p in geometry)
#         ncols = np.shape(geometry)[0] * 2
#         ntimepoints =  np.shape(data)[0]
#         matrix =self.getwaveformspacetime(data,geometry,yrange)
#                    
#         ims = []
#         for timepoint in range(ntimepoints):
#             
#             ##basic, no interpolation
#             if ax:
#             	probeimage = ax.imshow(ndimage.filters.gaussian_filter(matrix[:,:,timepoint],0.85),cmap=cmap,vmin=-0.3,vmax=1.0,aspect=aspect)
#             else:
#             	probeimage = plt.imshow(ndimage.filters.gaussian_filter(matrix[:,:,timepoint],0.85),cmap=cmap,vmin=-0.3,vmax=1.0,aspect=aspect)
#             #probeimage.set_xlim(-0.5,20)
#             #2d interpolation
# #            t = transform.resize(matrix[:,:,timepoint],(nrows*5,ncols*5))
# #            index = 0            
# #            X = np.zeros(np.shape(t)[0]  * np.shape(t)[1]);
# #            Y = np.zeros(np.shape(t)[0]  * np.shape(t)[1]);
# #            Z = np.zeros(np.shape(t)[0]  * np.shape(t)[1]);    
# #            for column in range(np.shape(t)[1]):
# #                for row in range(np.shape(t)[0]):
# #                        X[index] = column 
# #                        Y[index] = row  
# #                        Z[index] = t[row][column]
# #                        index+= 1                           
# #            f = scipy.interpolate.interp2d(X,Y,Z,kind='linear',copy=True)
# #            probeimage = plt.imshow(f(np.unique(X),np.unique(Y)),cmap=cmap,aspect=.2)#vmin=-0.3,vmax=1.0,
# 
#             ims.append([probeimage])
#         
#         ani = animation.ArtistAnimation(self.fig,ims,interval=60,blit=False,repeat_delay=500)
#         #plt.show()
#         #ani.save('C:\Users\danield\Desktop\temp\149045\LH_1_2015-01-22_13-46-06_flashes_vis\graphics\allunits.mp4')
#         return ani
#         
#     def waveform_animateblobs(self,data,order,yrange=250,cmap=plt.cm.RdGy,geometry=np.array((np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2]))):
#         #same as waveform_animateblob, but for every unit in a dataset
#         #input: spikes dictionary
#         nrows = max(len(p) for p in geometry)
#         ncols = np.shape(geometry)[0]# * 4
#         ntimepoints = np.shape(data[data['units'][0]]['waveform'])[0]
#         allunits = np.zeros((nrows,len(order)  * ncols*2,ntimepoints))
#         
#         index = 0
#         for unit in order:
#             matrix = self.getwaveformspacetime(data[unit]['waveform'],geometry,yrange)
#             allunits[:,index:index+ncols,:]=matrix
#             index+=4   
#             
#         ims = []
#         for timepoint in range(ntimepoints):
#             
#             ##basic, no interpolation
#             probeimage = plt.imshow(ndimage.filters.gaussian_filter(allunits[:,:,timepoint],0.85),cmap=cmap,vmin=-0.3,vmax=1.0,aspect='auto')
#             #probeimage.set_xlim(-0.5,2)
#             ims.append([probeimage])
#         
#         ani = animation.ArtistAnimation(self.fig,ims,interval=120,blit=False,repeat_delay=500)
#         #ani.save('C:\Users\danield\Desktop\temp\149045\LH_1_2015-01-22_13-46-06_flashes_vis\graphics\allunits.mp4')
#         
#         plt.show()            
#         return ani
#         
# 	def waveform_animateblobs_sub(self,data,order,yrange=250,cmap=plt.cm.RdGy,geometry=np.array((np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2])),aspect='auto'):
# 		#same as waveform_animateblob, but for every unit in a dataset
# 		#input: spikes dictionary
# 		nrows = max(len(p) for p in geometry)
# 		ncols = np.shape(geometry)[0]# * 4
# 		ntimepoints = np.shape(data[data['units'][0]]['waveform'])[0]
# 		allunits = np.zeros((nrows,len(order)  * ncols*2,ntimepoints))
# 		nunits = len(order)
# 
# 		fig, axes = plt.subplots(1,nunits)
# 			
# 		index = 0
# 		for i,unit in enumerate(order):
# 
# 			matrix = self.getwaveformspacetime(data[unit]['waveform'],geometry,yrange)
# 			allunits[:,index:index+ncols,:]=matrix
# 			index+=4   
# 
# 			ims = []
# 			for timepoint in range(ntimepoints):
# 
# 			   ##basic, no interpolation
# 			   probeimage = axes[i].imshow(ndimage.filters.gaussian_filter(allunits[:,:,timepoint],0.85),cmap=cmap,vmin=-0.3,vmax=1.0,aspect='auto')
# 			   #probeimage.set_xlim(-0.5,2)
# 			   ims.append([probeimage])
# 		
# 		ani = animation.ArtistAnimation(fig,ims,interval=120,blit=False,repeat_delay=500)
# 		#ani.save('C:\Users\danield\Desktop\temp\149045\LH_1_2015-01-22_13-46-06_flashes_vis\graphics\allunits.mp4')
# 
# 		plt.show()            
# 		return ani
# 	
#     #plot spheres for the units over the probe
#     #draws the imec probe, based on the geometry, in the 3D view
#     #draws a sphere, colored for single units 
#     #input: 
#     #   data: dictionary of spikes, formatted like the output of openephys.load_kwik_klusteres()
#     #   geometry: probe geometry. for data save in 
#     #   sitespacing: distance between sites. tuple (x spacing,y spacing). default (22,22), for phase 2 imec probes.
#     #   multishank: (optional) if not -1, treat each entry in geometry as a shank with center-to-center spacing equal to multishank
#     #output: none. makes plot and makes it visible  
#     def units(self,data,geometry=0,sitespacing=(22,22),multishank=-1):
# 
#         if geometry == 0 :
#             geometry = self.waveformorder
#         if 'centroid' in data[data.keys()[1]]:
#             self.view.setCameraPosition(distance=4000)
#             scalefactor = 100
#     
#             #plot the probe back
#             x1=0;x2=49;y1=0;y2=1408;z1=0;z2=20  
#             m1 = self.rectanglemesh(0,49,0,1408,0,20,[0.2,0.2,0.2,0.9])
#             m1.setGLOptions('additive')
#             self.view.addItem(m1)
#             #add the sites
#             for row in range(64):
#                 for col in range(2):
#                     x1=3+22*col;x2=x1+18;y1=(22*row);y2=y1+18;z1=18;z2=22  
#                     m1 = self.rectanglemesh(x1,x2,y1,y2,z1,z2,[0.8,0.8,0.8,1])
#                     m1.setGLOptions('additive')
#                     self.view.addItem(m1)
#                         
#             #plot the spheres
#             for unit in data.keys():
#                 if type(data[unit]) == dict:
#                     if 'centroid' in data[unit].keys():
#                         centroid = data[unit]['centroid']
#                         sphere = gl.MeshData.sphere(rows=10,cols=30,radius=10.0)
#                         if 'type' in data[unit].keys():
#                             if data[unit]['type'] == 'MUA' or data[unit]['type'] == 'unit':
#                                 colors = np.ones((sphere.faceCount(), 4), dtype=float)
#                                 if data[unit]['type'] == 'MUA':
#                                     colors[:]=0.1;
#                                     colors[:,3] = 0.5
#                                 if data[unit]['type'] == 'unit':
#                                     colors[:,0]=np.random.uniform()
#                                     colors[:,1]=np.random.uniform()
#                                     colors[:,2]=np.random.uniform()
#                                     colors[:,3] = 0.5
#                                 sphere.setFaceColors(colors)
#                                 m3 = gl.GLMeshItem(meshdata=sphere, smooth=False)#, shader='balloon')
#                                 m3.translate(centroid[1]*sitespacing[0]+11,centroid[0]*sitespacing[1]+11,30,local=True)                
#                                 self.view.addItem(m3)
#             self.view.show()
#         else:
#             print('must compute centroid data for units before plotting units over probe')
#             print('run ephys.findcentroids() to find centroids')
#    
#     def hideunitsplot():
#         self.view.hide()
#     def showunitsplot():
#         self.view.show() 
#     def rectanglemesh(self,x1,x2,y1,y2,z1,z2,color):
#         #utility for 3D plotting of a rectangle [a.k.a. box, a.k.a. orthotope]
#         #generates a mesh for plotting with pyqtgraph's OpenGL functions 
#         #input: the coordinates of a box and and RBGA color
#         #   x1,x2,y1,y2,z1,z2 are ints    
#         #   color is list [R,B,G,alpha]
#         #output: pyqtgraph.opengl.glMeshItem which can be added to a gl view
#         verts = np.array([
#             [x1,y1,z1],
#             [x2,y1,z1],
#             [x1,y2,z1],
#             [x2,y2,z1],
#             [x1,y1,z2],
#             [x2,y1,z2],
#             [x1,y2,z2],
#             [x2,y2,z2]
#             ])
#         faces = np.array([
#             [0, 2, 6],
#             [0, 6, 4],
#             [2, 3, 6],
#             [6, 3, 7],
#             [1, 3, 7],
#             [1, 7, 5],
#             [0, 1, 4],
#             [1, 4, 5],
#             [4, 5, 6],
#             [5, 6, 7],
#             [0, 1, 2],
#             [1, 2, 3]
#         ])
#         colors = np.array([
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]],
#             [color[0], color[1], color[2], color[3]]
#         ])
#         m1 = gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False)
#         return m1    
# 
# 
# 
#         
# def waveform_animateblobs_sub(data,order,yrange=250,cmap=plt.cm.RdGy,geometry=np.array((np.linspace(1,128,128)[::2],np.linspace(1,128,128)[1::2])),aspect='auto'):
# 	#same as waveform_animateblob, but for every unit in a dataset
# 	#input: spikes dictionary
# 	nrows = max(len(p) for p in geometry)
# 	ncols = np.shape(geometry)[0] * 2
# 	ntimepoints = np.shape(data[data['units'][0]]['waveform'])[0]
# 	allunits = np.zeros((nrows,len(order)  * ncols*2,ntimepoints))
# 	nunits = len(order)
# 
# 	fig, axes = plt.subplots(1,nunits)
# 			
# 	index = 0
# 	for i,unit in enumerate(order):
# 
# 		matrix = getwaveformspacetime(data[unit]['waveform'],geometry,yrange)
# 		allunits[:,index:index+ncols,:]=matrix
# 		index+=4   
# 
# 		ims = []
# 		for timepoint in range(ntimepoints):
# 
# 		   ##basic, no interpolation
# 		   probeimage = axes[i].imshow(ndimage.filters.gaussian_filter(allunits[:,:,timepoint],0.85),cmap=cmap,vmin=-0.3,vmax=1.0,aspect='auto')
# 		   #probeimage.set_xlim(-0.5,2)
# 		   ims.append([probeimage])
# 		
# 	ani = animation.ArtistAnimation(fig,ims,interval=10,blit=False,repeat_delay=500)
# 	ani.save('C:\Users\danield\Desktop\temp\149045\LH_1_2015-01-22_13-46-06_flashes_vis\graphics\allunits.mp4')
# 
# 	plt.show()            
# 	return ani
