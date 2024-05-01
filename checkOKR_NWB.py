from pynwb import NWBHDF5IO
import h5py, glob, os
import numpy as np
from ellipse import LsqEllipse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import tqdm
from p_tqdm import p_map
import time
import pandas as pd
from pynwb import TimeSeries
from pynwb.behavior import PupilTracking
from hdmf.backends.hdf5.h5_utils import H5DataIO
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, TimeSeries, NWBHDF5IO, ProcessingModule
from pynwb.behavior import PupilTracking
from joblib import Parallel, delayed

def get_eye_frame_from_NWB(nwb,index):
    positions = ['DR','D','LD','L','RV','R','VL','V']
    x,y = [],[]
    for p in positions:
        pos = np.array(nwb.processing.get('behavior').get('PupilTracking')[p+'pupil_positions'].data)
        x.extend([pos[index,0]])
        y.extend([pos[index,1]])
    return x,y
def get_eye_frame(df,index):
    positions = ['DR','D','LD','L','RV','R','VL','V']
    x,y = [],[]
    for p in positions:
        x_ = df[p+'pupil_X'].values[index]
        y_ = df[p+'pupil_Y'].values[index]
        x.extend([x_])
        y.extend([y_])
    return x,y
def fit_ellipse(x,y):
    lsqe = LsqEllipse()
    reg = lsqe.fit( np.array(list(zip(x, y))))
    center, width, height, phi = reg.as_parameters()
    return center, width, height, phi

    
def check_OKR_NWB(recording_folder,
                 experimenter = 'j_h',
                 experiment_description= 'Denman Lab (Hughes Lab collaboration) on Autobahn Therepeutics remyelination therapy.'):
    if len(glob.glob(os.path.join(recording_folder,'*resnet*'))) > 2: 
        # print('making NWB for '+recording_folder)

            
        nwb_path = os.path.join('/Volumes/s1/autobahn/nwbs/okr/with_center',os.path.basename(recording_folder))+'_c.nwb'
        if os.path.exists(nwb_path): print('FOUND: '+recording_folder)
        else: print('MISSING: '+recording_folder)
    return None

#walk folder tree
rootDir = '/Volumes/s1/OKR/sessions'

for dirName, subdirList, fileList in os.walk(rootDir):
    if len(glob.glob(os.path.join(dirName,'*resnet*'))) > 2:
        if 'Cohort' in dirName and 'Prelim' not in dirName:# and '0' not in dirName:
            if len(glob.glob(os.path.join(dirName,'*resnet*'))) > 2:
                # print('Found directory: %s' % dirName)
                check_OKR_NWB(dirName) #do the work
            else: pass#print('NO DLC data for '+dirName)

# Parallel(n_jobs=7)(make_OKR_NWB(dirName) for dirName, subdirList, fileList in os.walk(rootDir)) #do the work)
# make_OKR_NWB('/Volumes/s1/OKR/sessions/Cohort1/Week1Data/3582(wk1.1of2)/3582(wk1.1of2)_2021_8_17-19_24_41')