#general imports
import os,sys,glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import csv 
import time, datetime
# ! pip install git+https://github.com/open-ephys/open-ephys-python-tools
from open_ephys.analysis import Session
from open_ephys.control import NetworkControl
import numpy as np
import matplotlib.pyplot as plt


def electrical_timestamps(path,rec_done,ec):
    
    #Pulls Wanted Data from directory location, turns it into session
    directory = path
    session = Session(directory)
    recordnode = session.recordnodes[0]
    data = session.recordnodes[0].recordings[0].continuous
    sampling_rate = 30000.
    analog_channels = data[0].samples
    times = data[0].timestamps/sampling_rate
    metadata = data[0].metadata  
    ##
    num_rec_done = list(range(rec_done))
    num_rec_done_a = np.array(num_rec_done)
    rec_num = rec_done
    # WRS
    w_high_samples = np.where(analog_channels[:,ec]>100)[0]
    w_low_samples = np.where(analog_channels[:,ec]<-100)[0]
    
    wrs = []
    wfs = []
    for transition in np.where(np.diff(w_high_samples)>10)[0]:
        wrs.extend([w_high_samples[transition]]) 
    for transition in np.where(np.diff(w_low_samples)>10)[0]:
        wfs.extend([w_low_samples[transition]+2])

    wrs_a = np.array(wrs)
    wfs_a = np.array(wfs)
     ###                                       Starting samples for each recording
    wrs_start_a = [wrs[0]]
    wfs_start_a = [wfs[0]]

    for transition in np.where(np.diff(wrs)>50000)[0]:
        wrs_start_a.extend([wrs[transition]])
    for transition in np.where(np.diff(wfs)>50000)[0]:
        wfs_start_a.extend([wfs[transition]])    
    #                                     Ending Samples for each Recording
    wrs_end_a = []
    wfs_end_a = []

    for transition in np.where(np.diff(wrs)>30000)[0]:
        wrs_end_a.extend([wrs[transition]])
    for transition in np.where(np.diff(wfs)>30000)[0]:
        wfs_end_a.extend([wfs[transition]])

    wrs_end_a.append(wrs[-1])
    wfs_end_a.append(wfs[-1])
    #                              Visualization only Each Recordings Starting sample and ending sampling
    wrs_s = [] 
    wfs_s = []

    for num in num_rec_done:
        if (num <= rec_num):
            wrs_s.extend([wrs.index(wrs_start_a[num_rec_done[num]])]) 
    for num in num_rec_done:
        if (num <= rec_num):
            wfs_s.extend([wfs.index(wfs_start_a[num_rec_done[num]])]) 
    ##
    wrs_s_a = np.array(wrs_s)
    wfs_s_a = np.array(wfs_s)
    ##
    wrs_e = []
    wfs_e = []

    for num in num_rec_done:
        if (num <= rec_num):
            wrs_e.extend([wrs.index(wrs_end_a[num_rec_done[num]])]) 
    for num in num_rec_done:
        if (num <= rec_num):
            wfs_e.extend([wfs.index(wfs_end_a[num_rec_done[num]])]) 

    wrs_e_a = np.array(wrs_e)
    wfs_e_a = np.array(wfs_e)
    #creating a list, in which ers_x[0] contains the entire length of recording 1, and ers_x[1] contains all of recording 2
    wrs_i = []
    wfs_i = []

    for num in num_rec_done:
        if (num <= rec_num):
            wrs_i.extend([[(wrs_s_a[num_rec_done[num]]),(wrs_e_a[num_rec_done[num]])]]) 
    for num in num_rec_done:
        if (num <= rec_num):
            wfs_i.extend([[(wfs_s_a[num_rec_done[num]]),(wfs_e_a[num_rec_done[num]])]]) 
    wrs_i_a = np.array(wrs_i)
    wfs_i_a = np.array(wfs_i)       

    wrs_x = []
    wfs_x = []

    for num in num_rec_done:
        if (num <= rec_num):
            wrs_x.extend([wrs_a[wrs_i_a[num][0]+1:wrs_i_a[num][1]+1]])
    for num in num_rec_done:
        if (num <= rec_num):
            wfs_x.extend([wfs_a[wfs_i_a[num][0]+1:wfs_i_a[num][1]+1]])
    
    # print(wrs_x[5])
    for num in num_rec_done:
        if (num <= rec_num):
            print('Recording ' + str(num+1))
            print('')
            print('wrs_x: ' + str(wrs_x[num][0]) +'   '+ str(wrs_x[num][-1])+'   '+str(np.shape(wrs_x[num]))+'   '+str(wfs_x[num][0]-wrs_x[num][0]))
            print('wfs_x: ' + str(wfs_x[num][0]) +'   '+ str(wfs_x[num][-1])+'   '+str(np.shape(wfs_x[num]))+'   '+str(wfs_x[num][-1]-wrs_x[num][-1]))
            print('')

def trigger_timestamps(path,rec_done,tc):
    
    #Pulls Wanted Data from directory location, turns it into session
    directory = path
    session = Session(directory)
    recordnode = session.recordnodes[0]
    data = session.recordnodes[0].recordings[0].continuous
    sampling_rate = 30000.
    analog_channels = data[0].samples
    times = data[0].timestamps/sampling_rate
    metadata = data[0].metadata  
    ##
    num_rec_done = list(range(rec_done))
    num_rec_done_a = np.array(num_rec_done)
    rec_num = rec_done
    # ERS
    np.shape(np.where(analog_channels[:,tc]>15000)[0]) 
    
    e_high_samples = np.where(analog_channels[:,tc]<15000)[0]
    e_low_samples = np.where(analog_channels[:,tc]>15000)[0]
  
    ers = []
    efs = []
    for transition in np.where(np.diff(e_high_samples)>10)[0]:
        ers.extend([e_high_samples[transition]])
    for transition in np.where(np.diff(e_low_samples)>10)[0]:
        efs.extend([e_low_samples[transition]])

    ers_a = np.array(ers)
    efs_a = np.array(efs)
    
    # Creating Lists Containing Starting sample # for each recording inside ers
    ers_start_a = [ers_a[0]] #first rising edge of reach recording ers_a[0]
    efs_start_a = [efs_a[0]] #first falling edge of each recording efs_a[0]

    for transition in np.where(np.diff(ers_a)>50000)[0]:
        ers_start_a.extend([ers_a[transition]])
    for transition in np.where(np.diff(efs_a)>50000)[0]:
        efs_start_a.extend([efs_a[transition]])    

    # Creating Lists Containing Ending Sample # for each Recording inside ers
    ers_end_a = [] #last rising edge of each recording
    efs_end_a = [] #last falling edge of each recording

    for transition in np.where(np.diff(ers_a)>30000)[0]:
        ers_end_a.extend([ers_a[transition]])
    for transition in np.where(np.diff(efs_a)>30000)[0]:
        efs_end_a.extend([efs_a[transition]])

    ers_end_a.append(ers_a[-2])
    efs_end_a.append(efs_a[-1])
    # Creating Lists Containing Starting index locations in ers for each recording
    ers_s = [] 
    efs_s = []

    for num in num_rec_done:
        if (num <= rec_num):
            ers_s.extend([ers.index(ers_start_a[num_rec_done[num]])]) 
    for num in num_rec_done:
        if (num <= rec_num):
            efs_s.extend([efs.index(efs_start_a[num_rec_done[num]])]) 

    ers_s_a = np.array(ers_s)
    efs_s_a = np.array(efs_s)
    # 
    ers_e = []
    efs_e = []

    for num in num_rec_done:
        if (num <= rec_num):
            ers_e.extend([ers.index(ers_end_a[num_rec_done[num]])]) 
    for num in num_rec_done:
        if (num <= rec_num):
            efs_e.extend([efs.index(efs_end_a[num_rec_done[num]])]) 

    ers_e_a = np.array(ers_e)
    efs_e_a = np.array(efs_e)
    # creating a list, in which ers_x[0] contains the entire length of recording 1, and ers_x[1] contains all of recording 2
    ers_i = []
    efs_i = []

    for num in num_rec_done:
        if (num <= rec_num):
            ers_i.extend([[(ers_s_a[num_rec_done[num]]),(ers_e_a[num_rec_done[num]])]]) 
    for num in num_rec_done:
        if (num <= rec_num):
            efs_i.extend([[(efs_s_a[num_rec_done[num]]),(efs_e_a[num_rec_done[num]])]]) 
    ers_i_a = np.array(ers_i)
    efs_i_a = np.array(efs_i)       

    ers_x = []
    efs_x = []

    for num in num_rec_done:
        if (num <= rec_num):
            ers_x.extend([ers_a[ers_i_a[num][0]+1:ers_i_a[num][1]+1]])  # [ers_a[ers_i_a[num][0]+1:ers_i_a[num][1]]])
    for num in num_rec_done:
        if (num <= rec_num):
            efs_x.extend([efs_a[efs_i_a[num][0]+1:efs_i_a[num][1]+1]])  # [efs_a[efs_i_a[num][0]+1:efs_i_a[num][1]]])
    # Print Statements
    print('ers & efs')
    print('')
    print('ers: ' + str(ers[0]) +'   '+ str(ers[-1])+ '   '+ str(np.shape(ers)))
    print('efs: ' + str(efs[0]) +'   '+ str(efs[-1])+'    '+ str(np.shape(efs)))
    print('')


    for num in num_rec_done:
        if (num <= rec_num):
            print('Recording ' + str(num+1))
            print('')
            print('ers_x: ' + str(ers_x[num][0]) +'   '+ str(ers_x[num][-1])+'   '+str(np.shape(ers_x[num]))+'   '+str(efs_x[num][0]-ers_x[num][0]))
            print('efs_x: ' + str(efs_x[num][0]) +'   '+ str(efs_x[num][-1])+'   '+str(np.shape(ers_x[num]))+'   '+str(efs_x[num][0]-ers_x[num][0]))
            print('')