import pickle as pkl
import numpy as np
import datetime as dt
import pandas as pd
import os, h5py, json,glob
import matplotlib.pyplot as plt
from tqdm import tqdm

#Convert meta file into dictionary 
#Adapted from readSGLX.py package from SpikeGLX
def readAPMeta(bin_path):
    metaPath = glob.glob(os.path.join(bin_path,'*ap.meta'))[0]
    metaName = os.path.basename(metaPath)
    metaDict = {}
    if os.path.isfile(metaPath):
        # print("meta file present")
        with open(metaPath) as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)

def readNIMeta(bin_path):
    print(bin_path)
    metaPath = glob.glob(os.path.join(bin_path,'*nidq.meta'))[0] #glob.glob(bin_path+'*nidq.meta')[0]
    metaName = os.path.basename(metaPath)
    metaDict = {}
    if os.path.isfile(metaPath):
        # print("meta file present")
        with open(metaPath) as f:
            mdatList = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdatList:
                csList = m.split(sep='=')
                if csList[0][0] == '~':
                    currKey = csList[0][1:len(csList[0])]
                else:
                    currKey = csList[0]
                metaDict.update({currKey: csList[1]})
    else:
        print("no meta file")
    return(metaDict)

#Takes the .bin file output from your SpikeGLX recording and outputs a plot of the individual 
# digital lines over time 
def parse_ni_digital(bin_path, seconds=True):
    print('Sit back. This is going to take a while!')
    #Memory map the bin file and parse into binary lines
    mm = np.memmap(glob.glob(os.path.join(bin_path, '*bin'))[0],dtype=np.int16) #glob.glob(bin_path+'*bin')[0],dtype=np.int16)
    digital_words = mm[8::9]
    
    #Extract the number of digital channels from the meta file
    meta = readNIMeta(bin_path)
    nchans = meta['niXDChans1']
    ncs = nchans.split(":")
    nChans = int(ncs[1])-int(ncs[0])+1
    
    #go through each timepoint and figure out if each line has switched states (high --> low or low --> high). 
    #the output will be two dictionaries digital_lines_rising and digital_lines_falling, 
    #which have each sample where the transition happened
    num_digital_channels=nChans 
    digital_lines_rising = {}
    digital_lines_falling = {}
    for i in tqdm(range(digital_words.shape[0])[::10]): #note that this downsamples by factor 10, to 100kHz
        if i==0:
            state_previous_sample = '{0:08b}'.format(digital_words[i]) #Parse digital words into binary lines. 1=High, 0=Low
            for line in range(num_digital_channels):
                digital_lines_rising['D'+str(line)] = [] #initialize empty list
                digital_lines_falling['D'+str(line)] = [] #initialize empty list
        else:
            state_this_sample = '{0:08b}'.format(digital_words[i]) #Parse digital words into binary lines. 1=High, 0=Low
            changes = [j for j in range(len(state_previous_sample)) if state_previous_sample[j] != state_this_sample[j]]
            for line in changes:   
                    if state_this_sample[line] == '1':
                        digital_lines_rising['D'+str(line)].extend([i*10]) #note that this scales back up to 1MHz sampling rate 
                    else:
                        digital_lines_falling['D'+str(line)].extend([i*10])  #note that this scales back up to 1MHz sampling rate 
            state_previous_sample=state_this_sample #update sample
            
    #Reorder the lines manually because they were set up backwards above
    #TODO: fix the above loop to put things in the right order the first time
    digital_lines_rising2 = {}
    digital_lines_falling2 = {}
    for i,key in enumerate(digital_lines_rising.keys()):
        digital_lines_rising2['D'+str(7-i)]=digital_lines_rising[key]
        digital_lines_falling2['D'+str(7-i)]=digital_lines_falling[key]
    digital_lines_rising = digital_lines_rising2
    digital_lines_falling = digital_lines_falling2
    if seconds==False:
        return(digital_lines_rising, digital_lines_falling) 

    #Convert from sample times to seconds
    if seconds==True:
        for line in digital_lines_rising.keys():
            digital_lines_rising[line] = np.array(digital_lines_rising[line])/float(meta['niSampRate'])
        for line in digital_lines_falling.keys():
            digital_lines_falling[line] = np.array(digital_lines_falling[line])/float(meta['niSampRate'])
        return(digital_lines_rising, digital_lines_falling) 

#plot the output of `sglx_nidaq`
def sglx_nidaq_plot(digital_lines_rising):
        for i,line in enumerate(digital_lines_rising.keys()):
            plt.plot(np.array(digital_lines_rising[line]),np.ones(len(digital_lines_rising[line]))*(7-i),'-o',label=line)
            plt.legend(loc='lower right')

#autoparse the output of sglx_nidaq into a dictionary of timestamps 
#SPECIFIC TO COLOR_POPULATION.PY
#requires 2 lists 1)stimulus names for line D1 in order 2)number of frames per stimulus in order
def cpop_autoparse(nidaq_dlr, d1frames, d1stims, d2frames, d2stims):
    #NIDAQ digital lines rising data will either come from a saved pkl file or a dictionary output of spikeglx_nidaq
    if isinstance(nidaq_dlr, str):
        with open(nidaq_dlr, 'rb') as a:
            nidaq = pkl.load(a)
    else:
        nidaq = nidaq_dlr
    #isolate the keys containing timestaps from lines D1 and D3
    d1 = np.array(nidaq['D1'])/1e7#ONLY FOR TESTING FIX BEFORE USING
    d2 = np.array(nidaq['D2'])/1e7#ONLY FOR TESTING FIX BEFORE USING
    
    #parse into dictionary
    stop=0
    stimulus_timestamps={}
    for i,j in zip(d1stims, enumerate(d1frames)):
        j=int(j[0])
        start = stop
        stop = start+d1frames[j]
        stim_ts = np.array(d1[start:stop])      
        stimulus_timestamps.update({str(i): stim_ts})

    for i,j in zip(d2stims, enumerate(d2frames)):
            j=int(j[0])
            start = stop
            stop = start+d2frames[j]
            stim_ts = np.array(d2[start:stop])      
            stimulus_timestamps.update({str(i): stim_ts})
 
    return(stimulus_timestamps)

#Takes pkl output from stimulus timestamps from cpop_autoparse to make 
#a dictionary with the stimulus data and timestamps. Needed to be a dictionary because of 3d arrays in stim data
#will dump pkl of dictionary in output path if save = true
def cpop_mtx_int(matrix_pkl, timestamps, output_path, save=True):
    if type(matrix_pkl)==dict:
        for key in matrix_pkl.keys():
            if key=='stackG':
                color_matrix_green = {'times':timestamps, 'frames':matrix_pkl[key]}
                if save==True:
                    pkl.dump(color_matrix_green,open(os.path.join(output_path, 'color_matrix_green.pkl'),'wb'))
            elif key=='stackB':
                color_matrix_uv = {'times':timestamps, 'frames':matrix_pkl[key]}
                if save==True:
                    pkl.dump(color_matrix_uv,open(os.path.join(output_path, 'color_matrix_uv.pkl'),'wb'))
        return(color_matrix_green, color_matrix_uv)
    else:
        highspeed_data = {'times':timestamps, 'frames':matrix_pkl}
        if save==True:
            pkl.dump(highspeed_data,open(os.path.join(output_path, 'highspeed.pkl'),'wb'))
        return(highspeed_data)

#returns a dataframe for each gratings stimulus
#unable to save because df.to_json doesnt work in function
#TODO: incorporate errors and dropped frames
#if youre working with gratings_color.py, ensure gratings_color=True
def cpop_gratings_int(gratings_pkl,timestamps,color_gratings=False):  
    if color_gratings==True:    
        gratings_df = pd.DataFrame(gratings_pkl['bgsweeptable'], 
                                   columns = ['Contrast', 'PosY', 'TF', 'SF', 'Phase', 'PosX', 'Ori', 'Color'])
        gratings_df['frame_no'] = gratings_df.index
        frameno_df = pd.DataFrame(gratings_pkl['bgsweeporder'], columns = ['frame_no'])
        gratings_df1 = pd.merge(left = frameno_df, 
                          right = gratings_df, 
                          on = 'frame_no', 
                          sort = False, 
                          how = 'left')
        gratings_df1[['R', 'G', 'B']] = pd.DataFrame(gratings_df1['Color'].tolist(), index=gratings_df1.index)   
        gratings_df2 = gratings_df1.drop(columns=['Color'])
        gratings_df2['times'] = timestamps
        return(gratings_df2)
        #if save==True:
            #filename = str(os.path.join(output_path, pkl_str+'.json'))
            #json.dump(gratings_df2, filename)
        return(gratings_df2)
    
    else:
        gratings_df = pd.DataFrame(gratings_pkl['bgsweeptable'], columns = ['contrast', 'posY', 'TF', 'SF', 'phase', 'posX', 'ori'])
        gratings_df['frame_no'] = gratings_df.index

        frameno_df = pd.DataFrame(gratings_pkl['bgsweeporder'], columns = ['frame_no'])

        gratings_df1 = pd.merge(left = frameno_df, 
                              right = gratings_df, 
                              on = 'frame_no', 
                              sort = False, 
                              how = 'left')
        gratings_df1['times'] = timestamps
        #print(gratings_df1.head())
        #if save==True:
            #filename = os.path.join(output_path, pkl_str+'.json')
            #gratings_df1.to_json()
        return(gratings_df1)

#Returns a dataframe for the scene flicker stimului like the matrix and gratings functions above
def cpop_scene_int(scene_pkl,timestamps):
    scene_files = []
    for i in range(len(scene_pkl['imagefiles'])):
        scene_files.append(os.path.basename(scene_pkl['imagefiles'][i])),

    scene_df = pd.DataFrame(scene_files, columns = ['Image File'])
    scene_df['frame_no'] = scene_df.index
    frameno_df = pd.DataFrame(scene_pkl['bgsweeporder'], columns = ['frame_no'])
    scene_df1 = pd.merge(left = frameno_df, 
                          right = scene_df, 
                          on = 'frame_no', 
                          sort = False, 
                          how = 'left')
    scene_df1.astype(dtype='S10')
    scene_df1['times'] = timestamps
    return(scene_df1)

#returns a data frame which contains all spike time data as well as the relevant info contained in cluster_info
#Highly recommend exporting the dataframe to .json
#dataPath is the path holding ALL probe folders
def unitTimes(dataPath,**sampling_rate):
    #get individual folders for each probe
    folder_paths = []
    imec0_path = glob.glob(dataPath+'*imec0')
    if len(imec0_path)!=0:
        folder_paths.append(imec0_path)
    imec1_path = glob.glob(dataPath+'*imec1')
    if len(imec1_path)!=0:
        folder_paths.append(imec1_path)        
    imec2_path = glob.glob(dataPath+'*imec2')
    if len(imec2_path)!=0:
        folder_paths.append(imec2_path)
    imec3_path = glob.glob(dataPath+'*imec3')
    if len(imec3_path)!=0:
        folder_paths.append(imec3_path)
    #print(folder_paths)    
    #get spike times and cluster groups
    unit_times = []
    for i, folder in enumerate(tqdm(folder_paths)):
        probe_names = ['imec0', 'imec1', 'imec2', 'imec3']
        # if not sampling_rate:
        #     imec_meta = readMeta(folder[0]+'\\') #extract meta file
        #     sampRate = float(imec_meta['imSampRate']) #get sampling rate (Hz)
        # else:
        #     sampRate  = float(sampling_rate['sampling_rate'])
        #cluster_groups = pd.read_csv(os.path.join(folder[0], 'cluster_group.tsv'), '\t') #redundant data found in cluster_info
        cluster_info = pd.read_csv(os.path.join(folder[0], 'cluster_info.tsv'), '\t')
        # spike_times = np.ndarray.flatten(np.load(os.path.join(folder[0], 'spike_secs.npy')))
        # spike_seconds = np.ndarray.flatten(spike_times/sampRate) #convert spike times to seconds from samples
        spike_seconds = np.ndarray.flatten(np.load(os.path.join(folder[0], 'spike_secs.npy')))
        spike_clusters = np.ndarray.flatten(np.load(os.path.join(folder[0], 'spike_clusters.npy')))
    
        #Generate Unit Times Table
        for index, unitID in enumerate(cluster_info['id'].values):
            unit_times.append({'probe':probe_names[i],
                               'unit_id': unitID,
                               'group': cluster_info.group[index],
                               'depth':cluster_info.depth[index],
                               'no_spikes': cluster_info.n_spikes[index], 
                               'amplitude':cluster_info.Amplitude[index],
                               'times': spike_seconds[spike_clusters == unitID],
                                })
    unit_data = pd.DataFrame(unit_times)
    #Remove clusters with no associated spike times left over from Phy
    for i,j in enumerate(unit_data.times):
        if len(unit_data.times[i])==0:
            unit_data.times[i]='empty'
    unit_times = unit_data[unit_data.times!='empty']
    return(unit_times)