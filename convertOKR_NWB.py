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

def make_OKR_NWB(recording_folder,
                 experimenter = 'j_h',
                 experiment_description= 'Denman Lab (Hughes Lab collaboration) on Autobahn Therepeutics remyelination therapy.'):
    if len(glob.glob(os.path.join(recording_folder,'*resnet*'))) > 2: 
        print('making NWB for '+recording_folder)
        try:
            #load data
            df_stimulus_frame_times = pd.read_csv(os.path.join(recording_folder,'stimulus_frame_times.csv'))
            df_stimulus_params = pd.read_csv(os.path.join(recording_folder,'stimulus_params.csv'))
            df_stimulus_times = pd.read_csv(os.path.join(recording_folder,'stimulus_times.csv'),index_col=False)

            df_stimulus_times['Orientation']=df_stimulus_params
            df_stimulus_times['TemporalFrequency']=[float(a[1:]) for a in df_stimulus_times.Value.values]
            df_stimulus_times.drop('Value', axis=1, inplace=True)

            df_camera_frames = pd.read_csv(os.path.join(recording_folder,'camera_times.csv'),index_col=False,header=None)
            df_dlc = pd.read_csv(glob.glob(os.path.join(recording_folder,'camera*0.csv'))[0],skiprows=2)
            df_dlc.columns=['frame','Lpupil_X','Lpupil_Y','Lpupil_likelihood','LDpupil_X','LDpupil_Y','LDpupil_likelihood',
                            'Dpupil_X','Dpupil_Y','Dpupil_likelihood','DRpupil_X','DRpupil_Y','DRpupil_likelihood',
                        'Rpupil_X','Rpupil_Y','Rpupil_likelihood','RVpupil_X','RVpupil_Y','RVpupil_likelihood',
                            'Vpupil_X','Vpupil_Y','Vpupil_likelihood','VLpupil_X','VLpupil_Y','VLpupil_likelihood','IRLED_X','IRLED_Y','IRLED_likelihood']
            df_dlc_filtered = pd.read_csv(glob.glob(os.path.join(recording_folder,'camera*_filtered.csv'))[0],skiprows=2)
            df_dlc_filtered.columns=['frame','Lpupil_X','Lpupil_Y','Lpupil_likelihood','LDpupil_X','LDpupil_Y','LDpupil_likelihood',
                            'Dpupil_X','Dpupil_Y','Dpupil_likelihood','DRpupil_X','DRpupil_Y','DRpupil_likelihood',
                        'Rpupil_X','Rpupil_Y','Rpupil_likelihood','RVpupil_X','RVpupil_Y','RVpupil_likelihood',
                            'Vpupil_X','Vpupil_Y','Vpupil_likelihood','VLpupil_X','VLpupil_Y','VLpupil_likelihood','IRLED_X','IRLED_Y','IRLED_likelihood']


            #create file
            nwbfile = NWBFile('Autobahn_myelin', 
                            recording_folder, 
                            datetime.now(tzlocal()),
                            experimenter=experimenter,
                            lab='Denman Lab',
                            institution='University of Colorado',
                            experiment_description=experiment_description,
                            session_id=os.path.basename(recording_folder))
            behavior_module = ProcessingModule('behavior', 'behavior module')
            nwbfile.add_processing_module(behavior_module)

            #add trials
            nwbfile.add_epoch(df_stimulus_times.Timestamp.iloc[0], 
                            df_stimulus_times.Timestamp.iloc[-1],
                            ['full_field_gratings'])
            nwbfile.add_trial_column('Orientation', 'the grating orientation')
            nwbfile.add_trial_column('TemporalFrequency', 'the grating temporal frequency')
            for i,row in df_stimulus_times.iterrows():
                nwbfile.add_trial(start_time=row.Timestamp,
                                stop_time=row.Timestamp+np.mean(np.diff(df_stimulus_times.Timestamp.values)),
                                Orientation=row.Orientation,
                                TemporalFrequency=row.TemporalFrequency)

            #add eye tracking
            eye_timestamps = df_camera_frames[0].values/1e9 - df_camera_frames[0].values[0]/1e9
            pupil = TimeSeries(
                name='IR_position',
                timestamps=eye_timestamps,
                data=np.ravel(np.column_stack((df_dlc.IRLED_X.values,df_dlc.IRLED_Y.values))).reshape(-1,2),
                unit='arb. unit',
                description='Features extracted from the video of the right eye.',
                comments='the IR LED position'
            )
            pupil_track = PupilTracking(pupil)
            for p in ['L','LD','D','DR','R','RV','V','VL']:
                eye_xy = TimeSeries(
                    name=p+'pupil_positions',
                    timestamps=eye_timestamps,
                    data=np.ravel(np.column_stack((df_dlc[p+'pupil_X'].values,df_dlc[p+'pupil_Y'].values))).reshape(-1,2),
                    unit='arb. unit',
                    description='Features extracted from the video of the right eye.',
                    comments='The 2D position of the center of this marker in the video '
                            'frame. This is not registered to degrees visual angle, but '
                            'could be used to detect saccades or other changes in eye position.'
                )
                pupil_track.add_timeseries(eye_xy)

                eye_xy_filtered = TimeSeries(
                    name=p+'pupil_positions_filtered',
                    timestamps=eye_timestamps,
                    data=np.ravel(np.column_stack((df_dlc_filtered[p+'pupil_X'].values,df_dlc_filtered[p+'pupil_Y'].values))).reshape(-1,2),
                    unit='arb. unit',
                    description='Features extracted from the video of the right eye.',
                    comments='The 2D position of the center of this marker in the video '
                            'frame. This is not registered to degrees visual angle, but '
                            'could be used to detect saccades or other changes in eye position.'   
                )
                pupil_track.add_timeseries(eye_xy_filtered)


            #do ellipse fitting
            num_frames = np.ravel(np.column_stack((df_dlc.IRLED_X.values,df_dlc.IRLED_Y.values))).reshape(-1,2).shape[0]
            center = np.zeros((num_frames,2))
            width= np.zeros(num_frames)
            height= np.zeros(num_frames)
            phi = np.zeros(num_frames)
            xdeg= np.zeros(num_frames)
            ydeg= np.zeros(num_frames)

            def do_ellipse_fit(i):
                try:
                    x,y = get_eye_frame(df_dlc_filtered,i)
                    (center,width,height,phi) = fit_ellipse(x,y)
                    return (i,center,width,height,phi)
                except: return (i,np.nan,np.nan,np.nan,np.nan)

            # pool = Pool()
            inds = np.arange(num_frames)
            res=p_map(do_ellipse_fit,inds,num_cpus=7)

            for i,cen, w, h, p in res:
                (center[i,:],width[i],height[i],phi[i]) = cen, w, h, p 
                r = (height[i]+width[i])/2.
                ydeg[i]=np.rad2deg(np.arccos(height[i]/r))
                xdeg[i]=np.rad2deg(np.arccos(r/width[i]))
            
            # pupil = nwb.processing['behavior'].data_interfaces['PupilTracking']

            center_xy = TimeSeries(
                name='pupil_center_positions',
                timestamps=eye_timestamps,
                data=np.array([center[:,0],center[:,1]]),
                unit='pixel',
                description='Features extracted from the video of the right eye.',
                comments='The 2D position of the center of the pupil in the video '
                        'frame. This is not registered to degrees visual angle, but '
                        'could be used to detect saccades or other changes in eye position.'
            )

            width_ = TimeSeries(
            name='pupil_width',
            timestamps=eye_timestamps,
            data=width,
            unit='pixel',
            description='Features extracted from the video of the right eye.',
            comments='The width of the pupil ellipse in the video '
                    'frame. This is not registered to degrees visual angle, but '
                    'could be used to detect saccades or other changes in eye position.'
            )

            height_ = TimeSeries(
            name='pupil_height',
            timestamps=eye_timestamps,
            data=height,
            unit='pixel',
            description='Features extracted from the video of the right eye.',
            comments='The height of the pupil ellipse in the video '
                    'frame. This is not registered to degrees visual angle, but '
                    'could be used to detect saccades or other changes in eye position.'
            )

            phi_ = TimeSeries(
            name='pupil_angle',
            timestamps=eye_timestamps,
            data=phi,
            unit='pixel',
            description='Features extracted from the video of the right eye.',
            comments='The rotation of the pupil ellipse in the video '
                    'frame. This is not registered to degrees visual angle, but '
                    'could be used to detect saccades or other changes in eye position.'
            )

            #add the ellipse fits
            pupil_track.add_timeseries(center_xy)
            pupil_track.add_timeseries(width_)
            pupil_track.add_timeseries(height_)
            pupil_track.add_timeseries(phi_)
            
            #close up the pupil_tracking module
            behavior_module.add_data_interface(pupil_track)

            nwb_path = os.path.join('/Users/danieldenman/Desktop/data/withCenter',os.path.basename(recording_folder))+'_c.nwb'
            with NWBHDF5IO(nwb_path, 'w') as io:
                io.write(nwbfile)
            print('SUCCEEDED making NWB for '+recording_folder)
        except: print('FAILED making NWB for '+dirName); return None
    return None

#walk folder tree
rootDir = '/Volumes/s1/OKR/sessions'

for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    if len(glob.glob(os.path.join(dirName,'*resnet*'))) > 2:
        if 'Cohort2' in dirName and 'Prelim' not in dirName:
            try:
                make_OKR_NWB(dirName) #do the work
            except: print('FAILED making NWB for '+dirName)
        else: print('NOT MAKING for '+dirName)

# Parallel(n_jobs=7)(make_OKR_NWB(dirName) for dirName, subdirList, fileList in os.walk(rootDir)) #do the work)
# make_OKR_NWB('/Volumes/s1/OKR/sessions/Cohort1/Week1Data/3582(wk1.1of2)/3582(wk1.1of2)_2021_8_17-19_24_41')