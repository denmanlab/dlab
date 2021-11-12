import numpy as np
import h5py, os, sys, glob
import pandas as pd
from hdmf.backends.hdf5.h5_utils import H5DataIO
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, TimeSeries, NWBHDF5IO, ProcessingModule
from pynwb.behavior import PupilTracking


def make_OKR_NWB(recording_folder,
                 experimenter = 'j_h',
                 experiment_description= 'Denman Lab (Hughes Lab collaboration) on Autobahn Therepeutics remyelination therapy.'):
    print('making NWB for '+recording_folder)
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
    behavior_module.add_data_interface(pupil_track)

    nwb_path = os.path.join(recording_folder,os.path.basename(recording_folder))+'.nwb'
    with NWBHDF5IO(nwb_path, 'w') as io:
        io.write(nwbfile)
    print('SUCCEEDED making NWB for '+recording_folder)

#walk folder tree
rootDir = '/Users/danieljdenman/data/autobahn'
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    if len(glob.glob(os.path.join(dirName,'*resnet*'))) > 2:
        try:
            make_OKR_NWB(dirName) #do the work
        except: print('FAILED making NWB for '+dirName)