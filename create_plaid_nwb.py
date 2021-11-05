import numpy as np
import matplotlib.pyplot as plt
import h5py, os, sys, glob, tqdm, datetime
import _pickle as pkl
import pandas as pd

import pynwb
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.utils import StrDataset
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.device import Device
from pynwb.epoch import TimeIntervals
from pynwb.ecephys import ElectrodeGroup
from pynwb.behavior import BehavioralEvents, BehavioralEpochs, BehavioralTimeSeries, PupilTracking, IntervalSeries


from dateutil.tz import tzlocal
from pynwb import NWBFile


#class data object made by dan for our task. 
#input when making a new instance is a path (string) to the data in the Deepnote file tree
#e.g., s = session('/data/thisisafolder/thisisthefolderthecsvfilesarein')
class Session:
    def __init__(self,path):
        self.path = path.lower()
        self.df_pulls = pd.read_csv(os.path.join(path,'lever_pulls.csv'))
        self.df_pushes = pd.read_csv(os.path.join(path,'lever_pushes.csv'))
        self.df_releases = pd.read_csv(os.path.join(path,'lever_releases.csv'))
        self.df_rewards = pd.read_csv(os.path.join(path,'rewards.csv'),index_col=False)
        self.df_target_images = pd.read_csv(os.path.join(path,'target_images.csv'))
        self.df_trial_ends = pd.read_csv(os.path.join(path,'trial_ends.csv'))
        self.df_trial_starts = pd.read_csv(os.path.join(path,'trial_starts.csv'))

        self.df_pulls['lever_direction'] =  ['pull' for i in range(self.df_pulls.shape[0])]
        self.df_pushes['lever_direction'] =  ['push' for i in range(self.df_pushes.shape[0])]
        self.df_lever = pd.concat((self.df_pushes,self.df_pulls),ignore_index=True).sort_values('Frame')

    def clean_duplicates(self, df):
        return df.drop(np.argwhere(np.diff(df.Frame.values) < 5)[0][0]+1)

    def check_number_of_trials(self):
        if self.df_trial_starts.shape[0] == self.df_lever.shape[0] == self.df_releases.shape[0] == (self.df_target_images.shape[0]+1):
            self.num_trials = self.df_trial_starts.shape[0]
            print('number of trials: '+str(self.num_trials))
            return True
        else:
            # print('number of trial starts: '+str(self.df_trial_starts.shape[0]))
            # print('number of lever moves: '+str(self.df_lever.shape[0]))
            # print('number of releases: '+str(self.df_releases.shape[0]))
            # print('number of target images: '+str((self.df_target_images.shape[0]+1)))
            return False
    
    def make_df(self, phase, offset=0):
        if phase == 1:
            if not self.check_number_of_trials():
                
                if self.df_trial_starts.shape[0] == self.df_lever.shape[0]:
                    #make sure releases match by removing releases that happen within 3 frames of each other
                    self.df_releases = self.df_releases.iloc[np.where(np.diff(self.df_releases.Frame.values)>3)[0]]
                else: 
                    print(path+' trial starts and lever movements do not match, which is a problem for Phase1')
                    return

            #enforce target image shape
            self.df_target_images=self.df_target_images.iloc[np.where(self.df_target_images.Frame.values < np.max(self.df_trial_starts.Frame.values))[0]]
            
            #combine times for each trial
            self.df = pd.concat([self.df_trial_starts.drop('Value',axis=1).reset_index(),
                                self.df_releases.drop('Value',axis=1).reset_index(),
                            self.df_lever.drop('Value',axis=1).reset_index()],
                            axis=1,ignore_index=False)

            #rename columns
            self.df.columns = ['trial_start_index','trial_start_frame','trial_start_timestamp',
                        'release_index','release_frame', 'release_timestamp',
                        'lever_move_index','lever_move_frame','lever_move_timestamp','lever_move_direction']

            #put in target and possible reward information and times
            # reward_time = np.zeros(self.df.shape[0]);reward_time[:]=np.nan
            # for reward_frame in self.df_rewards.Frame.values:
            #     for trial in range(self.df.shape[0]):
            #         if abs(reward_frame - self.df['release_frame'][trial]) < 4:
            #             reward_time[trial] = self.df_rewards.Timestamp[self.df_rewards.Frame==reward_frame]
            # self.df['reward_timestamp']=reward_time
            self.df['target_image_int'] = [1]+np.array(self.df_target_images.Value).tolist()
            directions = ['','pull','push']
            image = ['cheetah','elephant']
            self.df['target_image'] = [image[i] for i in self.df['target_image_int']]
            self.df['target_direction'] = [directions[i] for i in self.df['target_image_int']]
        
            reward_,reward_times,reward_frames=[],[],[]
            for index, row in self.df.iterrows():
                start_frame =  self.df.trial_start_frame[index]
                try: end_frame = self.df.trial_start_frame[index+1]
                except:end_frame = self.df.trial_start_frame[index]+1000
                _ = gather_events_by_trial(self.df_rewards,start_frame,end_frame)
                reward_.extend([_[0]])
                reward_frames.extend([_[1]])
                reward_times.extend([_[2]])
            self.df['reward'] = reward_
            self.df['reward_times'] = reward_times
            self.df['reward_frames'] = reward_frames

        if phase == 2:
            self.df = self.df_trial_starts.drop('Value',axis=1)
            self.df.columns=['trial_start_time','trial_start_Timestamp']
            first_trial = self.df.trial_start_time.iloc[0]
            ind = np.where(self.df_trial_ends.Timestamp.values > first_trial)[0][0]
            trials_no = self.df.shape[0]
            print(trials_no)

            if self.df.trial_start_time.values[0] > self.df_trial_ends.Timestamp.values[0]:
                #find where the first df_trial_ends row is greater than the first trial
                self.df_trial_ends[self.df_trial_ends.Timestamp > first_trial]

            if trials_no > self.df_trial_ends.Frame.values[ind:].shape[0]:
                print('more starts than ends')
                self.end_frames = self.df_trial_ends.Frame.values[ind:].tolist()
                self.end_times = self.df_trial_ends.Timestamp.values[ind:].tolist()
                self.end_Timestamps = self.df_trial_ends.Value.values[ind:].tolist()
                print(np.shape(self.end_frames))
                for j in range(trials_no - np.shape(self.end_frames)[0]):
                    self.end_frames.extend([np.nan])
                    self.end_times.extend([np.nan])
                    self.end_Timestamps.extend([np.nan])
                print(np.shape(self.end_frames))
                self.df['trial_end_frame'] = self.end_frames
                self.df['trial_end_time'] = self.end_times
                self.df['trial_end_Timestamp'] =  self.end_Timestamps
            else:
                self.df['trial_end_frame'] = self.df_trial_ends.Frame.values[ind:]
                self.df['trial_end_time'] = self.df_trial_ends.Timestamp.values[ind:]
                self.df['trial_end_Timestamp'] = self.df_trial_ends.Value.values[ind:]
            image = ['cheetah','elephant']
            self.df['target_image'] = [image[i] for i in self.df_target_images.Value.values]

            pull_,push_,reward_ = [],[],[]
            pull_times,push_times,reward_times= [],[],[]
            push_frames,pull_frames,reward_frames= [],[],[]
            for index, row in self.df.iterrows():
                start_frame = index
                # end_frame = self.df.trial_end_frame[index+offset]
                try:
                    end_frame = self.df_trial_ends[self.df_trial_ends.Frame > start_frame].Frame.values[0]+2
                    #gather pulls
                    _ = gather_events_by_trial(self.df_pulls,start_frame,end_frame)
                    pull_.extend([_[0]])
                    pull_frames.extend([_[1]])
                    pull_times.extend([_[2]])
                    _ = gather_events_by_trial(self.df_pushes,start_frame,end_frame)
                    push_.extend([_[0]])
                    push_frames.extend([_[1]])
                    push_times.extend([_[2]])
                    _ = gather_events_by_trial(self.df_rewards,start_frame,end_frame)
                    reward_.extend([_[0]])
                    reward_frames.extend([_[1]])
                    reward_times.extend([_[2]])
                except:
                    _=[np.nan,np.nan,np.nan]
                    pull_.extend([_[0]])
                    pull_frames.extend([_[1]])
                    pull_times.extend([_[2]])
                    push_.extend([_[0]])
                    push_frames.extend([_[1]])
                    push_times.extend([_[2]])
                    _=[False,[],[]]
                    reward_.extend([_[0]])
                    reward_frames.extend([_[1]])
                    reward_times.extend([_[2]])
            self.df['pulled'] = pull_
            self.df['pushed'] = push_
            self.df['rewarded'] = reward_
            self.df['pull_times'] = pull_times
            self.df['push_times'] = push_times
            self.df['reward_times'] = reward_times
            self.df['pull_frames'] = pull_frames
            self.df['push_frames'] = push_frames
            self.df['reward_frames'] = reward_frames
            lever_move_time = []
            for trial in self.df.iterrows():
                if np.isnan(trial[1].pulled) and np.isnan(trial[1].pushed):
                    lever_move_time.extend([np.nan])
                else:
                    if trial[1].pulled:
                        if type(trial[1].pull_times) == list:
                            lever_move_time.extend([trial[1].pull_times[0]])
                        else: lever_move_time.extend([trial[1].pull_times[0]])
                    else:
                        lever_move_time.extend([trial[1].push_times[0]])
            self.df['lever_move_time']=lever_move_time

            self.df['reaction_time'] = self.df.lever_move_time - self.df.trial_start_time

        if phase > 0:
            print(self.path)
            self.mouse = self.path.split('/')[-2].lower()
            self.date = self.path.split('/')[-1]
            # dt_string= self.date.strip(self.mouse)[1:]
            dt_string = self.date[4:]
            print(self.date)
            format = "%Y_%m_%d-%H_%M_%S"
            dt_object = datetime.datetime.strptime(dt_string, format)
            self.df['phase'] = [phase]*self.df.shape[0]
            self.df['datetime'] = [dt_object]*self.df.shape[0]
            self.df['year'] = [dt_object.year]*self.df.shape[0]
            self.df['month'] = [dt_object.month]*self.df.shape[0]
            self.df['day'] = [dt_object.day]*self.df.shape[0]
            self.start_in_seconds = dt_object.hour*60*60 + dt_object.minute*60 + dt_object.second 
            self.df['start_in_seconds'] = [self.start_in_seconds]*self.df.shape[0]
            try:
                for col in self.df.columns:
                    if 'time' in col:
                        if 'trial' in col:
                            self.df[col] =  np.array(self.df[col].values) + self.tart_in_seconds
                        else:
                            self.df[col] = [np.array(np.array(ts)+self.start_in_seconds).tolist() for ts in self.df[col].values]
                        # self.df[col].values+start_in_seconds
            except: pass
            try: self.df['target_image'] = [image[i] for i in self.df['target_image_int']]
            except: pass
            self.df['mouse']=[self.mouse]*self.df.shape[0]
        
def gather_events_by_trial(df,start_frame,end_frame):
#this function takes a DataFrame as input, and searches this DataFrame 
#for any events in the Frame column that falls between the input start_frame and end_frame 
    moved, frames, times = False, [], []
    for index, row_ in df.iterrows():
        if row_.Frame > start_frame and row_.Frame < end_frame:
            moved = True
            frames.extend([row_.Frame])
            times.extend([row_.Timestamp])
    return moved, frames, times

def check_session_phase(s):
    if s.df_trial_starts.shape[0] > 0: #make sure there are some trials 
        if s.df_trial_starts.shape[0] == s.df_lever.shape[0]:# in phase 1, trials are started by lever moves, so this is equal
            #could also be 2c, which is 2AFC! need to check
            return 1
        else:
            #TODO: differentiate between 2a and 2b and 3
            return 2
    else: return 0


def combine_Sessions(sessions):
#input: sessions is a list of Session objects created by dlabbehavior cheetah_elephant Session
#outpu
    sessions[0].make_df(2)
    df = sessions[0].df
    for s in sessions[1:]:
        s.make_df(2)
        df_toadd = s.df
        end_of_last_session_t = df.trial_end_time.values[-1]
        end_of_last_session_f = df.trial_end_frame.values[-1]
        for col in ['trial_start_time','trial_end_time','lever_move_time']:
            df_toadd[col] = df_toadd[col] + end_of_last_session_t
        
        for col in ['pull_times','push_times','reward_times']:
            df_toadd[col] = [np.array(np.array(times) + end_of_last_session_t).tolist for times in df_toadd[col]]
        
        df_toadd.trial_end_frame = df_toadd.trial_end_frame.values + end_of_last_session_f
        
        for col in ['pull_frames','push_frames','reward_frames']:
            df_toadd[col] = [np.array(np.array(frames) + end_of_last_session_f).tolist for frames in df_toadd[col]]
        
        df = pd.concat([df,df_toadd],ignore_index=True)
    return df

def sort_day_sessions(paths):
    new_names = []
    for path in paths:
        session_time_string = os.path.basename(path).split('-')[-1]
        if len(session_time_string.split('_')[0]) < 2:
            hour = '0'+ session_time_string.split('_')[0]
        else: hour = session_time_string.split('_')[0]
        if len(session_time_string.split('_')[1]) < 2:
            minute = '0'+ session_time_string.split('_')[1]
        else: minute = session_time_string.split('_')[1]
        if len(session_time_string.split('_')[2]) < 2:
            second = '0'+ session_time_string.split('_')[2]
        else: second = session_time_string.split('_')[2]
#os.path.basename(path).split('-')[0]+'-'+
        new_names.extend([hour+'_'+minute+'_'+second])
    return np.array(paths)[np.argsort(new_names).astype(int)]


def create_nwb(recording_folder):
    s = Session(recording_folder)

    s.make_df(2)#<--- change t

    nwb_file = NWBFile('Denman Lab plaid discrimination', 
                    recording_folder, 
                    datetime.datetime.now(tzlocal()),
                    experimenter=experimenter,
                    lab='Denman Lab',
                    institution='University of Colorado',
                    experiment_description=experiment_description,
                    session_id=os.path.basename(recording_folder))

    behavior_module = ProcessingModule('behavior', 'behavior module')
    nwb_file.add_processing_module(behavior_module)

    nwb_file.add_trial_column('image','image presented on this trial.')
    nwb_file.add_trial_column('imageApercent','contrast of one grating presented on this trial.')
    nwb_file.add_trial_column('imageBpercent','contrast of one grating presented on this trial.')
    nwb_file.add_trial_column('pulled','time in seconds that the trial ended.')
    nwb_file.add_trial_column('pushed','time in seconds that the trial ended.')
    nwb_file.add_trial_column('response_time','time in seconds that the trial ended.')
    nwb_file.add_trial_column('rewarded','if a reward was delivered on this trial')

    for i,unit_row in s.df.iterrows():
        if unit_row.target_image == 'elephant':
            percA= 0; percB= 100
        else: 
            percA= 100; percB= 0
        nwb_file.add_trial(start_time=unit_row.trial_start_time,
                        stop_time=unit_row.trial_end_time,
                        image='grating',
                        imageApercent=str(percA),
                        imageBpercent=str(percB),
                        pulled=str(unit_row.pulled),
                        pushed=str(unit_row.pushed),
                        response_time=str(unit_row.lever_move_time),
                        rewared=unit_row.rewarded)
    try: 
        lf = pd.read_csv(os.path.join(recording_folder,'buffered_lever_frames.csv'),index_col=False)
        val_ = [int(vala[1:])for vala in lf.Value]
        lf.Value = val_
        
        lever_ts = TimeSeries(
            name='lever_position',
            timestamps=lf.Timestamp.values,
            data=lf.Value.values,
            unit='arbitrary',
            description='The position reading of the 4 way encoder attached to the lever the mouse pushes forward, backward, left and right with his forelimbs.',
            comments='Lever movement. 512 is centered.'
        )
        nwb_file.add_acquisition(lever_ts)
    except:
        print('failed to add lever data to '+recording_folder)

    nwb_path = os.path.join(recording_folder,os.path.basename(recording_folder))+'_plaid.nwb'
    with pynwb.NWBHDF5IO(nwb_path, 'w') as io:
        io.write(nwb_file)


if __main__:
    recording_folder = '/Users/danieljdenman/Desktop/c70/c70_2021_10_30-17_3_12'
    experimenter = 'djd'
    experiment_description= 'Denman Lab plaid direction discrimination'
