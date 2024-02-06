#general imports
import os,sys,glob
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
plt.style.use('dark_background')
import seaborn as sns
import csv 
import time, datetime

from pynwb import NWBFile
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.device import Device
from pynwb.epoch import TimeIntervals
from pynwb.ecephys import ElectrodeGroup
from pynwb.behavior import BehavioralEvents, BehavioralEpochs, BehavioralTimeSeries, PupilTracking, IntervalSeries
   
from dateutil.tz import tzlocal
sns.set_palette(sns.color_palette('Set2')[3:])


class Session:
    """#class data object made by dan for our task. 
        #input when making a new instance is a path (string) to the data in the Deepnote file tree
        #e.g., s = session('/data/thisisafolder/thisisthefolderthecsvfilesarein')
    """
    def __init__(self,path):
        self.path = path.lower()
        print(self.path)
        self.df_pulls = pd.read_csv(os.path.join(path,'lever_pulls.csv'))
        self.df_pushes = pd.read_csv(os.path.join(path,'lever_pushes.csv'))
        self.df_releases = pd.read_csv(os.path.join(path,'lever_releases.csv'))
        self.df_rewards = pd.read_csv(os.path.join(path,'rewards.csv'),index_col=False)
        self.df_target_images = pd.read_csv(os.path.join(path,'target_images.csv'))
        self.df_trial_ends = pd.read_csv(os.path.join(path,'trial_ends.csv'))
        self.df_trial_starts = pd.read_csv(os.path.join(path,'trial_starts.csv'))
        if os.path.exists(os.path.join(path,'orientation.csv')):
            self.df_orientations = pd.read_csv(os.path.join(path,'orientation.csv'))

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
            first_trial = self.df.trial_start_Timestamp.iloc[0]
            ind = np.where(first_trial > self.df_trial_ends.Timestamp.values)
            trials_no = self.df.shape[0]

            
            if self.df.trial_start_time.values[0] > self.df_trial_ends.Timestamp.values[0]:
            #     #find where the first df_trial_ends row is greater than the first trial
                # print('first start: '+str(first_trial))
                self.df_trial_ends = self.df_trial_ends[self.df_trial_ends.Timestamp > first_trial]
            
            
            if trials_no > self.df_trial_ends.shape[0]:
                print('more starts than ends')
                self.end_frames = self.df_trial_ends.Frame.values[trials_no*-1:].tolist()
                self.end_times = self.df_trial_ends.Timestamp.values[trials_no*-1:].tolist()
                self.end_Timestamps = self.df_trial_ends.Timestamp.values[trials_no*-1:].tolist() #self.df_trial_ends.Value.values[trials_no*-1:].tolist()
                for j in range(trials_no - self.df_trial_ends.shape[0]):
                    self.end_frames.extend([np.nan])
                    self.end_times.extend([np.nan])
                    self.end_Timestamps.extend([np.nan])
                self.df['trial_end_frame'] = self.end_frames
                # self.df['trial_end_time'] = self.end_times
                self.df['trial_end_Timestamp'] =  self.end_Timestamps
                self.df['trial_end_time'] = self.end_Timestamps
            else:
                self.df['trial_end_frame'] = self.df_trial_ends.Frame.values[trials_no*-1:]
                # self.df['trial_end_time'] = self.df_trial_ends.Value.values[trials_no*-1:]
                self.df['trial_end_Timestamp'] = self.df_trial_ends.Timestamp.values[trials_no*-1:]
                self.df['trial_end_time'] = self.df_trial_ends.Timestamp.values[trials_no*-1:]
            image = ['cheetah','elephant']
            
            self.df['target_image'] = [image[i] for i in self.df_target_images.Value.values]
            
            self.df['trial_start_time'] = self.df['trial_start_Timestamp']

            pull_,push_,reward_ = [],[],[]
            pull_times,push_times,reward_times= [],[],[]
            push_frames,pull_frames,reward_frames= [],[],[]
            for index, row in self.df.iterrows():
                start_frame = self.df_trial_starts.Frame.iloc[index]
                # end_frame = self.df.trial_end_frame[index+offset]
                try:
                    # end_frame = 
                    end_frame = self.df_trial_ends[self.df_trial_ends.Frame > start_frame].Frame.values[0]+2
                    #gather pulls
                    _ = gather_events_by_trial(self.df_pulls,start_frame,end_frame)
                    # print(start_frame)
                    # print(end_frame)
                    # print(_)
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
            self.mouse = self.path.split('/')[-2].lower()
            self.date = self.path.split('/')[-1]
            # dt_string= self.date.strip(self.mouse)[1:]
            dt_string = self.date[4:]
            format = "%Y_%m_%d-%H_%M_%S"
            dt_object = datetime.datetime.strptime(dt_string, format)
            self.df['phase'] = [phase]*self.df.shape[0]
            self.df['datetime'] = [dt_object]*self.df.shape[0]
            self.df['year'] = [dt_object.year]*self.df.shape[0]
            self.df['month'] = [dt_object.month]*self.df.shape[0]
            self.df['day'] = [dt_object.day]*self.df.shape[0]
            start_in_seconds = dt_object.hour*60*60 + dt_object.minute*60 + dt_object.second 
            self.df['start_in_seconds'] = [start_in_seconds]*self.df.shape[0]
            try:
                for col in self.df.columns:
                    if 'time' in col:
                        if 'trial' in col:
                            self.df[col] =  np.array(self.df[col].values) + start_in_seconds
                        else:
                            self.df[col] = [np.array(np.array(ts)+start_in_seconds).tolist() for ts in self.df[col].values]
                        # self.df[col].values+start_in_seconds
            except: pass
            try: self.df['target_image'] = [image[i] for i in self.df['target_image_int']]
            except: pass
            self.df['mouse']=[self.mouse]*self.df.shape[0]

            if os.path.exists(os.path.join(path,'orientation.csv')):
                self.df['orientations'] = [int(v.strip(')')) for v in s.df_orientations.Value.values]
        
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
    
def gather_events_by_time(df,start_time,end_time):
#this function takes a DataFrame as input, and searches this DataFrame 
#for any events in the Frame column that falls between the input start_frame and end_frame 
    moved, frames, times, values = False, [], [], []
    for index, row_ in df.iterrows():
        if row_.Timestamp > start_time and row_.Timestamp < end_time:
            moved = True
            frames.extend([row_.Frame])
            times.extend([row_.Timestamp])
            values.extend([row_.Value])
    return moved, times, values

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

def import_timestamps(filename):
    #returns all timestamp columns as numpy arrays
    #input: path to csv with timestamps in a column called 'Timestamp'
    df1 = pd.read_csv(filename) 
    df = pd.to_numeric(df1['Timestamp'])                                  #These are actually timestamps
    timestamps = np.array(df)                                             #should be in seconds
    return timestamps

def make_df(path,rew_amt=-1):
    #Reads all timestamps from each csv from a session
    # filename = os.path.join(path,"trial_starts.csv")                  #should be when images came on
    # df_trial_starts = pd.read_csv(filename,index_col=False) 
    # trial_starts = np.array(df_trial_starts.Timestamp)
    
    filename = os.path.join(path,"lever_pulls.csv")
    df_lever_pulls = pd.read_csv(filename,index_col=False) 
    lever_pulls =  np.array(df_lever_pulls.Timestamp)
    
    filename = os.path.join(path, "lever_releases.csv")
    df_lever_releases = pd.read_csv(filename,index_col=False) 
    lever_releases =  np.array(df_lever_releases.Timestamp)

    filename = os.path.join(path, "trial_ends.csv")
    df_trial_ends = pd.read_csv(filename,index_col=False) 
    trial_ends =  np.array(df_trial_ends.Timestamp) 

    filename = os.path.join(path, "manual_rewards.csv")
    df_manual_rewards = pd.read_csv(filename,index_col=False) 
    manual_rewards = np.array(df_manual_rewards.Timestamp) 

    filename = os.path.join(path, "rewards.csv")
    df_rewards = pd.read_csv(filename,index_col=False) 
    df_rewards_withAmount = pd.read_csv(filename) 
    df_rewards['Value']=[int(amt.strip(')')) for amt in df_rewards_withAmount.Value]
    rewards = np.array(df_manual_rewards.Timestamp)

    filename = os.path.join(path, "target_times.csv")
    if os.path.exists(filename):
        df_target_times = pd.read_csv(filename,index_col=False) 

    ###Not Global Variables
    new_rew_amt = rew_amt

    ##################Makes df
    session_start = lever_pulls[0] #timestamp of first pull

    rewardsZero = rewards - session_start - .002

    lever_pullsZero = lever_pulls - session_start
    lever_releasesZero = lever_releases - session_start
    trial_endsZero = trial_ends - session_start
    # trial_startsZero = trial_starts - session_start

    filename = os.path.join(path, "target_times.csv")
    if os.path.exists(filename):
        target_timestamp = df_target_times.Timestamp - session_start
        a=df_target_times.Value[:-1].to_list()
        a.insert(0,1.0)
        target_times = a
    else:
        target_timestamp = [np.nan]*np.shape(lever_pulls)[0]
        target_times = target_timestamp
 
    if len(manual_rewards):
        manual_rewardsZero = manual_rewards - session_start 

    #Checking for alignment
    new_lever_pullsZero = lever_pullsZero
    new_lever_pullsZero1 = new_lever_pullsZero
    if len(lever_releasesZero) == len(lever_pullsZero) and np.min(lever_releasesZero-lever_pullsZero)>0:
        # plt.plot(lever_releasesZero-lever_pullsZero)
        #no alignment necessary
        new_lever_releasesZero = lever_releasesZero
        new_lever_releasesZero1 = lever_releasesZero
    else: 
        #aligment here, removes last lever pull if the session ends before release happens 
        new_lever_releasesZero = []
        for pull in lever_pullsZero:
            pullfound=False
            for release in lever_releasesZero: 
                if not pullfound:
                    if release > pull:
                        new_lever_releasesZero.extend([release])
                        pullfound=True

        if np.shape(new_lever_releasesZero)[0] + 1 == np.shape(new_lever_pullsZero)[0]:
            new_lever_releasesZero.extend([np.nan])
        new_lever_releasesZero1 = new_lever_releasesZero

    #Use these for properly aligned pulls/releases per trial:
    new_lever_pullsZeroArray = np.array(new_lever_pullsZero)  
    new_lever_releasesZeroArray = new_lever_releasesZero1

    #calculate the hold time
    actual_holdTime = new_lever_releasesZeroArray -  new_lever_pullsZeroArray

    #get the rewarded trials
    rewarded = [gather_events_by_time(df_rewards,start+session_start,end+session_start+0.200)[0] for start,end in zip(new_lever_pullsZeroArray,new_lever_releasesZeroArray)]

    # get the reward amounts
    # Keep in mind that only the rewarded lines will have the vol of reward
    reward_amount = []
    for start,end in zip(new_lever_pullsZeroArray,new_lever_releasesZeroArray):
        val = gather_events_by_time(df_rewards,start+session_start,end+session_start+0.200)[2]
        if len(val)>0: val = sum(val) #there was a reward, note the value 
        else: val = 0. #there was no reward, note the reward amount as 0
        reward_amount.extend([val])


    #for phase 1 and 2, create dummy targets
    if np.shape(target_timestamp)[0]==np.shape(rewarded)[0]:pass
    else:
        target_timestamp = [np.nan]*np.shape(rewarded)[0]
        target_times = target_timestamp

    #%% Construct the df:
    all_data = list(zip(new_lever_pullsZero, new_lever_releasesZero, actual_holdTime, rewarded, reward_amount,target_timestamp,target_times)) #, dip, p_dip))
    all_data_df = pd.DataFrame(all_data, columns=['pull_times', 'release_times', 'hold_time', 'rewarded','reward_amount','target_timestamp','target_hold_time'])#, 'dip cutoff', 'p_dip'])
    all_data_df['reaction_time']=all_data_df.hold_time-all_data_df.target_hold_time

    return all_data_df.iloc[1:]#throw out the first trial

def make_lever_NWB(recording_folder,experimenter='djd',experiment_description= 'Denman Lab lever visual engagement task',reward_vol=-1):
    nwb_path = os.path.join(recording_folder,os.path.basename(recording_folder))+'_lever.nwb'
    
    if os.path.exists(os.path.join(recording_folder,'image_ends.csv')):
        df_target = pd.read_csv(os.path.join(recording_folder,'target_times.csv'))
        if df_target.shape[0]<1 : phase = 2
        else: phase = 3
    else:
        phase=1

    if not os.path.exists(nwb_path):
        try:
            df = make_df(recording_folder)
        except:
            print('FAILED: '+recording_folder+' at df creation')
            return np.nan
        try:
            nwb_file = NWBFile(experiment_description, 
                    recording_folder, 
                    datetime.datetime.now(tzlocal()),
                    experimenter=experimenter,
                    lab='Denman Lab',
                    institution='University of Colorado',
                    experiment_description=experiment_description+' | phase:'+str(phase),
                    session_id=os.path.basename(recording_folder))
        except:
            print('FAILED: '+recording_folder+' at file creation')
            return np.nan
        try:
            behavior_module = ProcessingModule('behavior', 'behavior module')
            nwb_file.add_processing_module(behavior_module)
        except:
            print('FAILED: '+recording_folder+' at behavior module creation')
            return np.nan
        try:
            nwb_file.add_trial_column('hold_time','time in seconds the mouse held the lever for')
            nwb_file.add_trial_column('target_time','time in seconds that the mouse needed to hold the lever for.')
            nwb_file.add_trial_column('rewarded','if a reward was delivered on this trial')
            nwb_file.add_trial_column('reward_amount','amount in uL of the reward')
            
            if not 'target_time' in df.columns:
                df['target_time']=[np.nan]*df.shape[0]
            for i,unit_row in df.iterrows():
                nwb_file.add_trial(start_time=unit_row.pull_times,
                                stop_time=float(unit_row.release_times),
                                hold_time=unit_row.hold_time,
                                target_time=unit_row.target_time,
                                reward_amount=unit_row.reward_amount,
                                rewarded=unit_row.rewarded)
        except:
            print('FAILED: '+recording_folder+' at adding trials')
            return np.nan
        try:
            l = pd.read_csv(os.path.join(recording_folder,'lever.csv'),skiprows=1,header=None,index_col=False)
            start = pd.read_csv(os.path.join(recording_folder,'session_start.csv'),header=None,index_col=False)
            start_time = start.iloc[0].values[0].split('T')[-1].split('-')[0]
            l['counter'] = np.array([int(count[-13:-2]) if type(count)==str else np.nan for count in l[6]])/1e7
            l['time'] = [str(int(l[3].iloc[i]))+':'+str(int(l[4].iloc[i]))+':'+str(int(l[5].iloc[i]))+'.'+str(l[2].iloc[i])[2:] if not np.isnan(l[5].iloc[i]) else np.nan for i in range(l.shape[0])]
            lever_start = float(l.time.iloc[0].split(':')[-1]) - float(start_time.split(':')[-1])
            lever_offset = l['counter'].iloc[0] - lever_start
            l['Timestamp'] = l['counter'].values - lever_offset
            
            val_ = [int(vala[1:]) if  type(vala)==str else np.nan for vala in l[0]]
            l['Value'] = val_

            lever_ts = TimeSeries(
                name='lever_position',
                timestamps=l['counter'].values - lever_offset,
                data=val_,
                unit='arbitrary',
                description='The position reading of the 4 way encoder attached to the lever the mouse pushes forward, backward, left and right with his forelimbs.',
                comments='Lever movement. 512 is centered.'
            )
            nwb_file.add_acquisition(lever_ts)
        except: print('FAILED: '+recording_folder+' at lever data')
        try:
            with NWBHDF5IO(nwb_path, 'w') as io:
                io.write(nwb_file)
            print('SUCCEEDED: '+recording_folder)

        except:print('FAILED: '+recording_folder+' at writing')
    else: print('EXISTS: '+os.path.join('/root/work/nwbs_lever',os.path.basename(recording_folder))+'_lever.nwb')

def make_lever_NWB_verbose(recording_folder,experimenter='djd',experiment_description= 'Denman Lab lever visual engagement task',reward_vol=-1):
    if not os.path.exists(os.path.join(recording_folder,os.path.basename(recording_folder))+'_lever.nwb'):

        df = make_df(recording_folder,reward_vol)

        nwb_file = NWBFile(experiment_description, 
                recording_folder, 
                datetime.datetime.now(tzlocal()),
                experimenter=experimenter,
                lab='Denman Lab',
                institution='University of Colorado',
                experiment_description=experiment_description,
                session_id=os.path.basename(recording_folder))

        behavior_module = ProcessingModule('behavior', 'behavior module')
        nwb_file.add_processing_module(behavior_module)

        nwb_file.add_trial_column('hold_time','time in seconds the mouse held the lever for')
        nwb_file.add_trial_column('target_time','time in seconds that the mouse needed to hold the lever for.')
        nwb_file.add_trial_column('rewarded','if a reward was delivered on this trial')
        nwb_file.add_trial_column('reward_amount','amount in uL of the reward')
        
        if not 'target_time' in df.columns:
            df['target_time']=[np.nan]*df.shape[0]
        for i,unit_row in df.iterrows():
            nwb_file.add_trial(start_time=unit_row.pull_times,
                            stop_time=unit_row.release_times,
                            hold_time=unit_row.hold_time,
                            target_time=unit_row.target_time,
                            reward_amount=unit_row.hold_time,
                            rewarded=unit_row.rewarded)

        l = pd.read_csv(os.path.join(recording_folder,'lever.csv'),skiprows=1,header=None,index_col=False)
        start = pd.read_csv(os.path.join(recording_folder,'session_start.csv'),header=None,index_col=False)
        start_time = start.iloc[0].values[0].split('T')[-1].split('-')[0]
        l['counter'] = np.array([int(count[-13:-2]) for count in l[6]])/1e7
        l['time'] = [str(l[3].iloc[i])+':'+str(l[4].iloc[i])+':'+str(l[5].iloc[i])+'.'+str(l[2].iloc[i])[2:] for i in range(l.shape[0])]
        lever_start = float(l.time.iloc[0].split(':')[-1]) - float(start_time.split(':')[-1])
        lever_offset = l['counter'].iloc[0] - lever_start
        l['Timestamp'] = l['counter'].values - lever_offset
        
        val_ = [int(vala[1:])for vala in l[0]]
        l['Value'] = val_
        
        lever_ts = TimeSeries(
            name='lever_position',
            timestamps=l['counter'].values - lever_offset,
            data=val_,
            unit='arbitrary',
            description='The position reading of the 4 way encoder attached to the lever the mouse pushes forward, backward, left and right with his forelimbs.',
            comments='Lever movement. 512 is centered.'
        )
        nwb_file.add_acquisition(lever_ts)

        nwb_path = os.path.join('nwbs_lever',os.path.basename(recording_folder))+'_lever.nwb'
        with NWBHDF5IO(nwb_path, 'w') as io:
            io.write(nwb_file)
        print('SUCCEEDED: '+recording_folder)

def make_discrimination_NWB(recording_folder,experimenter='djd',experiment_description= 'Denman Lab visual engagement task',reward_vol=-1):
    nwb_path = os.path.join('/root/work/nwbs',os.path.basename(recording_folder))+'_ce.nwb'
    if not os.path.exists(nwb_path):
        print('making discimination NWB for '+recording_folder)
        try:
            s = Session(recording_folder)
            s.make_df(2)
            if s.df.trial_start_Timestamp.iloc[0] > s.df.trial_end_Timestamp.iloc[0]:
                print('FAILED: '+recording_folder+' at df alignment')
                print(s.df.trial_start_time.iloc[0]);print(s.df.trial_end_time.iloc[0])

        except:
            print('FAILED: '+recording_folder+' at df creation')
        try:
            # y,m,d = folders[0].split('/')[-1].split('-')[0].split('_')[1:] 
            # if int(y) == 2021 and int(m) > 9 and int(d) > 13: experiment_description= 'Denman Lab plaid discrimination'
            # else: experiment_description = 'Denman Lab cheetah or elephant discrimination'
            nwb_file = NWBFile(experiment_description, 
                    recording_folder, 
                    datetime.datetime.now(tzlocal()),
                    experimenter=experimenter,
                    lab='Denman Lab',
                    institution='University of Colorado',
                    experiment_description=experiment_description,
                    session_id=os.path.basename(recording_folder))
        except:print('FAILED: '+recording_folder+' at file creation')
        try:
            behavior_module = ProcessingModule('behavior', 'behavior module')
            nwb_file.add_processing_module(behavior_module)
        except:print('FAILED: '+recording_folder+' at behavior module creation')
        # try:s
        if 'cheetah' in experiment_description: 
            nwb_file.add_trial_column('image','image presented on this trial.')
            nwb_file.add_trial_column('percent_elephant','contrast of one grating presented on this trial.')
            nwb_file.add_trial_column('pulled','time in seconds that the trial ended.')
            nwb_file.add_trial_column('pushed','time in seconds that the trial ended.')
            nwb_file.add_trial_column('response_time','time in seconds that the trial ended.')
            nwb_file.add_trial_column('rewarded','if a reward was delivered on this trial')

            for i,unit_row in s.df.iterrows():
                if unit_row.target_image == 'elephant':
                    percent_elephant= 100; 
                else: 
                    percent_elephant= 0; 
                nwb_file.add_trial(start_time=unit_row.trial_start_Timestamp,
                                stop_time=float(unit_row.trial_end_Timestamp),
                                image=unit_row.target_image,
                                percent_elephant=str(percent_elephant),
                                pulled=str(unit_row.pulled),
                                pushed=str(unit_row.pushed),
                                response_time=str(unit_row.lever_move_time),
                                rewarded=unit_row.rewarded)
        if 'plaid'in experiment_description:
            nwb_file.add_trial_column('image','image presented on this trial.')
            nwb_file.add_trial_column('imageApercent','contrast of one grating presented on this trial.')
            nwb_file.add_trial_column('imageBpercent','contrast of one grating presented on this trial.')
            nwb_file.add_trial_column('pulled','time in seconds that the trial ended.')
            nwb_file.add_trial_column('pushed','time in seconds that the trial ended.')
            nwb_file.add_trial_column('response_time','time in seconds that the trial ended.')
            nwb_file.add_trial_column('rewarded','if a reward was delivered on this trial')
            # if 'orientation' in s.df.columns:
            nwb_file.add_trial_column('orientation','orientation of grating')


            for i,unit_row in s.df.iterrows():
                if unit_row.target_image == 'elephant':
                    percA= 0; percB= 100;
                else: 
                    percA= 100; percB= 0;
                if 'orientation' in s.df.columns:
                    ori = unit_row.orientation
                else: ori=np.nan
                nwb_file.add_trial(start_time=float(unit_row.trial_start_Timestamp)+s.df.start_in_seconds.iloc[0],
                                stop_time=float(unit_row.trial_end_Timestamp)+s.df.start_in_seconds.iloc[0],
                                image='grating',
                                imageApercent=str(percA),
                                imageBpercent=str(percB),
                                pulled=str(unit_row.pulled),
                                pushed=str(unit_row.pushed),
                                response_time=str(unit_row.lever_move_time),
                                rewarded=unit_row.rewarded,
                                orientation=ori)



        # except:print('FAILED: '+recording_folder+' at adding trials')
        # try:
        lf = pd.read_csv(os.path.join(recording_folder,'buffered_lever_frames.csv'),index_col=False)

        val_ = [int(vala[1:]) if  type(vala)==str else np.nan for vala in lf.Value]
        lf.Value = val_
        
        lever_ts = TimeSeries(
            name='lever_position',
        #     starting_time=lf.Timestamp.iloc[0],
            timestamps=lf.Timestamp.values+s.df.start_in_seconds.iloc[0],
            data=lf.Value.values,
            unit='arbitrary',
            description='The position reading of the 4 way encoder attached to the lever the mouse pushes forward, backward, left and right with his forelimbs.',
            comments='Lever movement. 512 is centered.'
        )
        nwb_file.add_acquisition(lever_ts)
        # except: 
        #     print('FAILED: '+recording_folder+' at lever data')
        try:
            nwb_path = os.path.join('/root/work/nwbs',os.path.basename(recording_folder))+'_ce.nwb'
            with NWBHDF5IO(nwb_path, 'w') as io:
                io.write(nwb_file)
            print('SUCCEEDED: '+recording_folder)
        except:print('FAILED: '+recording_folder+' at writing')

def make_behavior_NWB(recording_folder,experimenter,experiment_description,reward_vol=-1):
    if 'plaid' or 'cheetah' in experiment_description:
        make_discrimination_NWB(recording_folder,experimenter,experiment_description,reward_vol)
    if 'lever' in experiment_description:
        make_lever_NWB(recording_folder,experimenter,experiment_description)

def combine_nwb_sessions(paths):
    io = NWBHDF5IO(paths[0], mode='r')
    nwb_ = io.read()
    df = nwb_.trials.to_dataframe()
    for nwb_path in paths[1:]:
        io = NWBHDF5IO(nwb_path, mode='r')
        nwb_ = io.read()
        df_toadd = nwb_.trials.to_dataframe()
        end_of_last_session_t = df.stop_time.values[-1]+10. #offset 10 seconds by default. this is inaccurate but a minimum estimate
        for col in ['start_time','stop_time','response_time']:
            if col in df_toadd.columns:
                df_toadd[col] = df_toadd[col].values.astype(float) + end_of_last_session_t        
        df = pd.concat([df,df_toadd],ignore_index=True)
    return df

def across_session_plots_lever(df):
    fig = plt.figure(figsize=(8.5,11))
    ax_num_trials = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.05,.03),yspan=(0.1,0.34))
    ax_rewards = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.37,0.62),yspan=(0.1,0.34))
    ax_hold_times_hist = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.05,0.95),yspan=(0.4,0.95))
    ax_percent_correct = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.67,0.92),yspan=(0.1,0.34))

    num_trials,rewards, reward_vol,percent_correct = [],[],[],[]
    for session_ in df.session_id.unique():
        df_sess = df[df.session_id==session_]
        num_trials.extend([df_sess.shape[0]])
        rewards.extend([sum(df_sess.rewarded)])
        reward_vol.extend([sum(df_sess.reward_amount)])
        percent_correct.extend([sum(df_sess.rewarded) / float(df_sess.shape[0])])

        ax=sns.histplot(data=df_sess, x="hold_time",hue='rewarded',binrange=(0,12),bins=60,
                    ax=ax_hold_times_hist)

    ax_num_trials.plot(df.session_id.unique(),num_trials)  
    ax_rewards.plot(df.session_id.unique(),rewards,label='rewards',color=sns.color_palette[3])  
    ax_rewards.twinx().plot(df.session_id.unique(),reward_vol,label='reward volume',color=sns.color_palette[3])   
    ax_rewards.legend(bbox_to_anchor= (0.2, -0.2) )
    ax_percent_correct.plot(df.session_id.unique(),percent_correct)  
    
    return fig,ax

def generate_session_lever(mouse,date,return_=None,session='combine', nwb=None):
    if nwb!=None:
        paths=[nwb]
        io = NWBHDF5IO(nwb, mode='r')
        nwb_ = io.read()
        df = nwb_.trials.to_dataframe()
    else:
        nwb_folder_paths = glob.glob(os.path.join(r'C:\Users\denma\Desktop\bonsai_levertask\data',mouse,mouse+'_'+date+'*'))
        paths = [glob.glob(os.path.join(p,'*.nwb'))[0] for p in nwb_folder_paths]
        if len(paths)> 1:
            if session == 'combine':
                paths = sort_day_sessions(paths)
                df = combine_nwb_sessions(paths)
                io = NWBHDF5IO(paths[-1], mode='r')
                nwb_ = io.read()
            else:
                nwb_path = paths[session]
                io = NWBHDF5IO(nwb_path, mode='r')
                nwb_ = io.read()
                df = nwb_.trials.to_dataframe()
        else: 
            nwb_path = paths[0]
            io = NWBHDF5IO(nwb_path, mode='r')
            nwb_ = io.read()
            df = nwb_.trials.to_dataframe()

    if return_ == 'df':
        df['mouse_id']   = [mouse]*df.shape[0]
        df['session_id'] = [date]*df.shape[0]
        df['phase'] = [int(nwb_.experiment_description.split('phase')[1][1:])]*df.shape[0]
        return df

    if return_ == 'fig':
        fig = plt.figure(figsize=(8.5,11))
        ax_lever_all = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.0,1.0),yspan=(0,0.17))
        ax_lever_ex1 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.0,0.2),yspan=(0.24,0.34))
        ax_lever_ex2 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.26,0.46),yspan=(0.24,0.34))
        ax_lever_start = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.52,0.72),yspan=(0.24,0.34))
        ax_lever_end = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.78,1.0),yspan=(0.24,0.34))        
        ax_hold_times_hist = placeAxesOnGrid(fig,dim=(1,1),xspan=(0,0.25),yspan=(0.4,0.62))
        ax_hold_times_box1 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.35,0.6),yspan=(0.4,0.62))
        ax_hold_times_box2 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.7,0.95),yspan=(0.4,0.62))
        ax_hold_times_session = placeAxesOnGrid(fig,dim=(1,1),xspan=(0,0.42),yspan=(0.7,0.85))
        ax_hold_times_rolling = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.56,0.99),yspan=(0.7,0.85))
        ax_hold_times_rewards = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.56,0.99),yspan=(0.9,0.99))
        
        #plot lever - whole session
        if len(paths)> 1:
            if session == 'combine':
                paths = sort_day_sessions(paths)
                io = NWBHDF5IO(paths[0], mode='r')
                nwb_ = io.read()
                ax_lever_all.plot(nwb_.acquisition['lever_position'].timestamps,
                        nwb_.acquisition['lever_position'].data,'w')
                ax_lever_all.set_ylabel('lever position')
                ax_lever_all.set_xlabel('sesssion time (sec)')    
                end_time = nwb_.acquisition['lever_position'].timestamps[-1]
                for path in paths[1:]:
                    io = NWBHDF5IO(path, mode='r')
                    nwb_ = io.read()
                    ax_lever_all.plot(nwb_.acquisition['lever_position'].timestamps+end_time,
                            nwb_.acquisition['lever_position'].data,'w')
                    ax_lever_all.set_ylabel('lever position')
                    ax_lever_all.set_xlabel('sesssion time (sec)')   
                    end_time += nwb_.acquisition['lever_position'].timestamps[-1]
        else:
            ax_lever_all.plot(nwb_.acquisition['lever_position'].timestamps,
                    nwb_.acquisition['lever_position'].data,'w')
            ax_lever_all.set_ylabel('lever position')
            ax_lever_all.set_xlabel('sesssion time (sec)')
        
        #plot some zoomed in lever examples
        r = np.random.randint(np.array(nwb_.acquisition['lever_position'].timestamps).shape[0])
        ax_lever_ex2.plot(nwb_.acquisition['lever_position'].timestamps[r:r+180],
                    nwb_.acquisition['lever_position'].data[r:r+180],color='w')
        r = np.random.randint(np.array(nwb_.acquisition['lever_position'].timestamps).shape[0])
        ax_lever_ex1.plot(nwb_.acquisition['lever_position'].timestamps[r:r+180],
                    nwb_.acquisition['lever_position'].data[r:r+180],color='w')

        #plot lever around pulls and rewards
        lever_at_pull   = []
        lever_at_release = []
        for i,row in df.iterrows():
            start_indices = np.where((np.array(nwb_.acquisition['lever_position'].timestamps) > row.start_time - 0.5) & ((np.array(nwb_.acquisition['lever_position'].timestamps) < row.start_time + 1.5)))
            # start_indices = np.arange(np.where(np.array(nwb_.acquisition['lever_position'].timestamps) > row.start_time)[0][0] - 30, np.where(np.array(nwb_.acquisition['lever_position'].timestamps) > row.start_time)[0][0] + 90)
            if len(nwb_.acquisition['lever_position'].data[start_indices]) > 1:
                lever_at_pull.append(nwb_.acquisition['lever_position'].data[start_indices])
                ax_lever_start.plot(np.array(nwb_.acquisition['lever_position'].timestamps[start_indices])-row.start_time,nwb_.acquisition['lever_position'].data[start_indices],color='#808080',lw=0.5)
            
            end_indices = np.where((np.array(nwb_.acquisition['lever_position'].timestamps) > row.stop_time - 0.5) & ((np.array(nwb_.acquisition['lever_position'].timestamps) < row.stop_time + 1.5)))
            # end_indices = np.arange(np.where(np.array(nwb_.acquisition['lever_position'].timestamps) > row.stop_time)[0][0] - 30, np.where(np.array(nwb_.acquisition['lever_position'].timestamps) > row.stop_time)[0][0] + 90)
            if len(nwb_.acquisition['lever_position'].data[end_indices]) > 1:
                ax_lever_end.plot(np.array(nwb_.acquisition['lever_position'].timestamps[end_indices])-row.stop_time,nwb_.acquisition['lever_position'].data[end_indices],color='#808080',lw=0.5)
                lever_at_release.append(nwb_.acquisition['lever_position'].data[end_indices])
        ax_lever_start.set_title('lever at start of trials');ax_lever_start.set_ylabel('lever position');ax_lever_start.set_xlabel('time (sec)')
        ax_lever_end.set_title('lever at end of trials');ax_lever_end.set_ylabel('lever position');ax_lever_end.set_xlabel('time (sec)')
    
        
        # histogram of hold times 
        sns.histplot(data=df, x="hold_time",hue='rewarded',binrange=(0,12),bins=60,
                    ax=ax_hold_times_hist)

        #box plots of hold times
        ax = sns.boxplot(data=df,y='hold_time',x='rewarded',ax=ax_hold_times_box1)
        ax = sns.stripplot(data=df,y='hold_time',x='rewarded',ax=ax_hold_times_box1)
        # ax_hold_times_box1.set_ylim(0,12)


        ax = sns.boxplot(data=df,y='hold_time',x='rewarded',ax=ax_hold_times_box2)
        ax = sns.stripplot(data=df,y='hold_time',x='rewarded',ax=ax_hold_times_box2)
        # ax_hold_times_box2.set_ylim(0.01,12)
        ax_hold_times_box2.set_yscale('log')

        ax = sns.regplot(x="start_time", y="hold_time", data=df[df.rewarded==False],ax=ax_hold_times_session)
        ax = sns.regplot(x="start_time", y="hold_time", data=df[df.rewarded==True],ax=ax_hold_times_session)
        # plt.ylim(0,4)
        ax_hold_times_session.set_xlabel('session time (secs)')
        ax_hold_times_session.set_ylabel('hold time (secs)')

        ax_hold_times_rolling.plot(df[df.rewarded==False].start_time,
        df[df.rewarded==False].hold_time.rolling(5).mean())
        ax_hold_times_rolling.plot(df[df.rewarded==True].start_time,
        df[df.rewarded==True].hold_time.rolling(5).mean())
        ax_hold_times_rolling.plot(df.start_time,df.hold_time.rolling(5).mean())
        ax_hold_times_rolling.set_xlabel('session time (secs)')
        ax_hold_times_rolling.set_ylabel('hold time (secs)')

        minutes = np.arange(int(df.start_time.values[-1]/60))
        rewards_min = []
        for i in minutes:
            df_ = df[df.rewarded==True]
            rewards_min.extend([df_[(df_.start_time > i*60.) & (df_.start_time < i*60.+60)].shape[0]])
        seconds = minutes * 60.
        ax_hold_times_rewards.plot(seconds,rewards_min)
        ax_hold_times_rewards.set_xlabel('session time (secs)')
        ax_hold_times_rewards.set_ylabel('rewards per minute')

        fig.text(0.02,0.99, mouse, fontsize=14)
        fig.text(0.08,0.99, date, fontsize=14)
        fig.text(0.02,0.96, 'total rewards: '+str(df[df.rewarded==True].shape[0]), fontsize=14)
        fig.text(0.02,0.93, 'mean hold time: '+str(np.nanmean(df.hold_time)), fontsize=14)
        fig.text(0.02,0.90, 'mean rewarded time: '+str(np.nanmean(df[df.rewarded==True].hold_time)), fontsize=14)


        return fig,ax

def generate_session_plaid(mouse,date,return_=None,session='combine',log_hist=True):
    nwb_folder_paths = glob.glob(os.path.join(r'C:\Users\denma\Desktop\cheetah_or_elephant\data',mouse,mouse+'_'+date+'*'))
    paths = [glob.glob(os.path.join(p,'*.nwb'))[0] for p in nwb_folder_paths]
    if session == 'combine':
        paths = sort_day_sessions(paths);path=paths[0]
        io = NWBHDF5IO(path, mode='r')
        nwb_ = io.read()
        df = nwb_.trials.to_dataframe()
        for path in paths[1:]:
            df_ = NWBHDF5IO(path, mode='r').read().trials.to_dataframe()
            df = pd.concat([df,df_,],ignore_index=True)
    else:
        if len(paths) == 1:
            path = paths[0]
        if type(session)==int:
            path=paths[session]
        else: 
            print('incorrect format of session; provide either an int index to a sorted list of this days sessions, or str(combine) to combine all of this days sessions.')
            return np.nan
        df = NWBHDF5IO(path, mode='r').read().trials.to_dataframe()

    df['reaction_time']=df.response_time.values.astype(float) - df.start_time.values.astype(float)
    df['target_image'] = ['A' if row.imageApercent > row.imageBpercent else 'B' for i,row in df.iterrows()]

    df['mouse_id']   = [mouse]*df.shape[0]
    df['session_id'] = [date]*df.shape[0]
    # df['phase'] = [int(nwb_.experiment_description.split('phase')[1][1:])]*df.shape[0]
   
    if return_ == 'df':
        return df

    if return_ == 'fig':
        fig = plt.figure(figsize=(8.5,11))
        ax_lever_all = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.0,1.0),yspan=(0,0.12))
        ax_lever_ex1 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.0,0.2),yspan=(0.18,0.3))
        ax_lever_ex2 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.26,0.46),yspan=(0.18,0.3))
        ax_lever_image = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.57,1.0),yspan=(0.2,0.35))
        ax_lever_response = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.57,1.0),yspan=(0.44,0.59))        
        ax_correct_rolling = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.,0.43),yspan=(0.39,0.59))
        ax_rt = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.,0.43),yspan=(0.6,0.8))
        ax_rt_hist = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.52,0.67),yspan=(0.66,0.8))
        ax_rt_violin = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.78,1.0),yspan=(0.66,0.8))

        if len(paths) == 1 or type(session)==int:
            nwb_ = NWBHDF5IO(path, mode='r').read()
            lever_timestamps = np.array(nwb_.acquisition['lever_position'].timestamps)
            lever_data = np.array(nwb_.acquisition['lever_position'].data)
        else:
            if session == 'combine':
                print('combining '+mouse+' sessions from '+date+'...')
                lever_timestamps = []
                lever_data =      []
                for path in paths:
                    nwb_ = NWBHDF5IO(path, mode='r').read()
                    lever_timestamps.extend(nwb_.acquisition['lever_position'].timestamps)
                    lever_data.extend(nwb_.acquisition['lever_position'].data)
                lever_timestamps = np.array(lever_timestamps)
                lever_data = np.array(lever_data)
            else:print('incorrect format of session; provide either an int index to a sorted list of this days sessions, or str(combine) to combine all of this days sessions.')
        ax_lever_all.plot(lever_timestamps,lever_data,'w')
        ax_lever_all.set_ylabel('lever position')
        ax_lever_all.set_xlabel('sesssion time (sec)')  
            
        # #plot some zoomed in lever examples
        r = np.random.randint(lever_timestamps.shape[0])
        ax_lever_ex2.plot(lever_timestamps[r:r+180],
                    lever_data[r:r+180],color='w')
        r = np.random.randint(lever_timestamps.shape[0])
        ax_lever_ex1.plot(lever_timestamps[r:r+180],
                    lever_data[r:r+180],color='w')
        ylim = 30
        for ax in (ax_lever_ex1,ax_lever_ex2): ax.set_ylim(512-ylim,512+ylim)

        # #plot lever around images A and B
        A_levers = np.zeros((df[df.imageApercent > df.imageBpercent].shape[0],180));al=0
        B_levers = np.zeros((df[df.imageBpercent > df.imageApercent].shape[0],180));bl=0
        A_images = np.zeros((df[df.imageApercent > df.imageBpercent].shape[0],180));ai=0
        B_images = np.zeros((df[df.imageBpercent > df.imageApercent].shape[0],180));bi=0
        for i,row in df.iterrows():
            try:
                lever_trigger = np.where((lever_timestamps > float(row.response_time)))[0][0]
                image_trigger = np.where((lever_timestamps > float(row.start_time)))[0][0]
                lever_response_d = lever_data[lever_trigger-120:lever_trigger+60] 
                lever_response_t = lever_timestamps[lever_trigger-120:lever_trigger+60] 
                lever_image_d = lever_data[image_trigger-30:image_trigger+150] 
                lever_image_t = lever_timestamps[image_trigger-30:image_trigger+150] 
                if row.imageApercent > row.imageBpercent:
                    ax_lever_image.plot(np.linspace(-0.5,2.5,180),lever_image_d,lw=0.5,color=sns.color_palette()[0],alpha=0.3)
                    A_levers[al,:]=lever_image_d; al+=1
                else:
                    ax_lever_image.plot(np.linspace(-0.5,2.5,180),lever_image_d,lw=0.5,color=sns.color_palette()[1],alpha=0.3)
                    B_levers[bl,:]=lever_image_d; bl+=1     
                
                if row.imageApercent > row.imageBpercent:
                    ax_lever_response.plot(np.linspace(-2,1,180),lever_response_d,lw=0.5,color=sns.color_palette()[0],alpha=0.3)
                    A_images[ai,:]=lever_response_d; ai+=1
                else:
                    ax_lever_response.plot(np.linspace(-2,1,180),lever_response_d,lw=0.5,color=sns.color_palette()[1],alpha=0.3)
                    B_images[bi,:]=lever_response_d; bi+=1
            except:
                if row.imageApercent > row.imageBpercent:
                    A_levers = A_levers[:-1,:]
                    A_images = A_images[:-1,:]
                else:
                    B_levers = B_levers[:-1,:]
                    B_images = B_images[:-1,:]
            
        ax_lever_image.plot(np.linspace(-0.5,2.5,180),np.mean(A_levers,axis=0),lw=2,color=sns.color_palette()[0])
        ax_lever_image.plot(np.linspace(-0.5,2.5,180),np.mean(B_levers,axis=0),lw=2,color=sns.color_palette()[1])
        ax_lever_response.plot(np.linspace(-2,1,180),np.mean(A_images,axis=0),lw=2,color=sns.color_palette()[0])
        ax_lever_response.plot(np.linspace(-2,1,180),np.mean(B_images,axis=0),lw=2,color=sns.color_palette()[1])
            
        ax_lever_image.set_title('aligned to image onset')
        ax_lever_response.set_title('aligned lever movement')
        for ax in [ax_lever_image,ax_lever_response]:
            ax.set_xlabel('time (sec)',fontsize=10)
            ax.set_ylabel('lever (arb)',fontsize=10)

        window = 10
        ax_correct_rolling.plot(df[df.imageApercent > df.imageBpercent].start_time,
        df[df.imageApercent > df.imageBpercent].rewarded.rolling(window).mean(),color=sns.color_palette()[0])
        ax_correct_rolling.plot(df[df.imageBpercent > df.imageApercent].start_time,
        df[df.imageBpercent > df.imageApercent].rewarded.rolling(window).mean(),color=sns.color_palette()[1])
        ax_correct_rolling.plot(df.start_time,df.rewarded.rolling(window).mean(),color=sns.color_palette()[2])
        ax_correct_rolling.set_xlabel('session time (secs)')
        ax_correct_rolling.set_ylabel('% corr ('+str(window)+' trial average)')
        ax_correct_rolling.axhline(0.5,ls='--')
        ax_correct_rolling.set_ylim(0,1)

        # scatter of reaction times 
        sns.scatterplot(data=df, x='start_time',y="reaction_time",hue='target_image',style = 'rewarded', style_order=[True,False],
                    ax=ax_rt,legend=False)

        # histogram of reaction times 
        sns.histplot(data=df, x="reaction_time",hue='target_image',binrange=(0,12),bins=60,
                    ax=ax_rt_hist,legend=False,log_scale=log_hist)

        v = sns.violinplot(data=df, x='rewarded',y="reaction_time",hue='target_image',split = True, 
                    ax=ax_rt_violin)
        v.legend(bbox_to_anchor= (0.2, -0.2) );
        
        ax_rt_violin.set_yscale('log')
        ax_rt.set_yscale('log')
        if log_hist:ax_rt_hist.set_xscale('log')
        ax_rt_hist.set_xlim(0.1,12)

        fig.text(0.02,0.99, mouse, fontsize=14)
        fig.text(0.08,0.99, date, fontsize=14)
        fig.text(0.02,0.96, 'total percent correct: '+str(df[df.rewarded==True].shape[0]/float(df.shape[0])), fontsize=14)
        fig.text(0.02,0.93, 'median reaction time: '+str(np.nanmedian(df.reaction_time)), fontsize=14)
        # fig.text(0.02,0.90, 'mean rewarded time: '+str(np.nanmean(df[df.rewarded==True].hold_time)), fontsize=14)
    return fig

def generate_session_cheetah(mouse,date,return_=None,session='combine',log_hist=True):
    paths = glob.glob('/root/work/nwbs/'+mouse+'_'+date+'*_ce.nwb')
    if session == 'combine':
        paths = sort_day_sessions(paths);path=paths[0]
        io = NWBHDF5IO(path, mode='r')
        nwb_ = io.read()
        df = nwb_.trials.to_dataframe()
        for path in paths[1:]:
            df_ = NWBHDF5IO(path, mode='r').read().trials.to_dataframe()
            df = pd.concat([df,df_,],ignore_index=True)
    else:
        if len(paths) == 1:
            path = paths[0]
        if type(session)==int:
            path=paths[session]
        else: 
            print('incorrect format of session; provide either an int index to a sorted list of this days sessions, or str(combine) to combine all of this days sessions.')
            return np.nan
        df = NWBHDF5IO(path, mode='r').read().trials.to_dataframe()
    df['reaction_time']=df.response_time.values.astype(float) - df.start_time.values.astype(float)
    df['target_image']=df.image
    if return_ == 'df':
        return df

    if return_ == 'fig':
        fig = plt.figure(figsize=(8.5,11))
        ax_lever_all = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.0,1.0),yspan=(0,0.12))
        ax_lever_ex1 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.0,0.2),yspan=(0.18,0.3))
        ax_lever_ex2 = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.26,0.46),yspan=(0.18,0.3))
        ax_lever_image = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.57,1.0),yspan=(0.2,0.35))
        ax_lever_response = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.57,1.0),yspan=(0.44,0.59))        
        ax_correct_rolling = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.,0.43),yspan=(0.39,0.59))
        ax_rt = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.,0.43),yspan=(0.6,0.8))
        ax_rt_hist = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.52,0.67),yspan=(0.66,0.8))
        ax_rt_violin = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.78,1.0),yspan=(0.66,0.8))

        #plot lever - whole session
        if len(paths) == 1 or type(session)==int:
            nwb_ = NWBHDF5IO(path, mode='r').read()
            lever_timestamps = np.array(nwb_.acquisition['lever_position'].timestamps)
            lever_data = np.array(nwb_.acquisition['lever_position'].data)
        else:
            if session == 'combine':
                print('combining '+mouse+' sessions from '+date+'...')
                lever_timestamps = []
                lever_data =      []
                for path in paths:
                    nwb_ = NWBHDF5IO(path, mode='r').read()
                    lever_timestamps.extend(nwb_.acquisition['lever_position'].timestamps)
                    lever_data.extend(nwb_.acquisition['lever_position'].data)
                lever_timestamps = np.array(lever_timestamps)
                lever_data = np.array(lever_data)
            else:print('incorrect format of session; provide either an int index to a sorted list of this days sessions, or str(combine) to combine all of this days sessions.')
        ax_lever_all.plot(lever_timestamps,lever_data,'w')
        ax_lever_all.set_ylabel('lever position')
        ax_lever_all.set_xlabel('sesssion time (sec)')  
        
        # #plot some zoomed in lever examples
        r = np.random.randint(np.array(nwb_.acquisition['lever_position'].timestamps).shape[0])
        ax_lever_ex2.plot(nwb_.acquisition['lever_position'].timestamps[r:r+180],
                    nwb_.acquisition['lever_position'].data[r:r+180],color='w')
        r = np.random.randint(np.array(nwb_.acquisition['lever_position'].timestamps).shape[0])
        ax_lever_ex1.plot(nwb_.acquisition['lever_position'].timestamps[r:r+180],
                    nwb_.acquisition['lever_position'].data[r:r+180],color='w')

        # #plot lever around images A and B
        A_levers = np.zeros((df[df.target_image == 'elephant'].shape[0],180));al=0
        B_levers = np.zeros((df[df.target_image == 'cheetah'].shape[0],180));bl=0
        A_images = np.zeros((df[df.target_image == 'elephant'].shape[0],180));ai=0
        B_images = np.zeros((df[df.target_image == 'cheetah'].shape[0],180));bi=0
        for i,row in df.iterrows():
            lever_trigger = np.where((np.array(nwb_.acquisition['lever_position'].timestamps) > float(row.response_time)))[0][0]
            image_trigger = np.where((np.array(nwb_.acquisition['lever_position'].timestamps) > float(row.start_time)))[0][0]
            lever_response_d = nwb_.acquisition['lever_position'].data[lever_trigger-120:lever_trigger+60] 
            lever_response_t = nwb_.acquisition['lever_position'].timestamps[lever_trigger-120:lever_trigger+60] 
            lever_image_d = nwb_.acquisition['lever_position'].data[image_trigger-30:image_trigger+150] 
            lever_image_t = nwb_.acquisition['lever_position'].timestamps[image_trigger-30:image_trigger+150] 
            if row.target_image =='elephant':
                ax_lever_image.plot(np.linspace(-0.5,2.5,180),lever_image_d,lw=0.5,color=sns.color_palette()[0],alpha=0.3)
                A_levers[al,:]=lever_image_d; al+=1
            else:
                ax_lever_image.plot(np.linspace(-0.5,2.5,180),lever_image_d,lw=0.5,color=sns.color_palette()[1],alpha=0.3)
                B_levers[bl,:]=lever_image_d; bl+=1
            
            if row.target_image =='elephant':
                ax_lever_response.plot(np.linspace(-2,1,180),lever_response_d,lw=0.5,color=sns.color_palette()[0],alpha=0.3)
                A_images[ai,:]=lever_response_d; ai+=1
            else:
                ax_lever_response.plot(np.linspace(-2,1,180),lever_response_d,lw=0.5,color=sns.color_palette()[1],alpha=0.3)
                B_images[bi,:]=lever_response_d; bi+=1
            
        ax_lever_image.plot(np.linspace(-0.5,2.5,180),np.mean(A_levers,axis=0),lw=2,color=sns.color_palette()[0])
        ax_lever_image.plot(np.linspace(-0.5,2.5,180),np.mean(B_levers,axis=0),lw=2,color=sns.color_palette()[1])
        ax_lever_response.plot(np.linspace(-2,1,180),np.mean(A_images,axis=0),lw=2,color=sns.color_palette()[0])
        ax_lever_response.plot(np.linspace(-2,1,180),np.mean(B_images,axis=0),lw=2,color=sns.color_palette()[1])
            
        ax_lever_image.set_title('aligned to image onset')
        ax_lever_response.set_title('aligned lever movement')
        for ax in [ax_lever_image,ax_lever_response]:
            ax.set_xlabel('time (sec)',fontsize=10)
            ax.set_ylabel('lever (arb)',fontsize=10)

        window = 10
        ax_correct_rolling.plot(df[df.target_image=='elephant'].start_time,
        df[df.target_image=='elephant'].rewarded.rolling(window).mean(),color=sns.color_palette()[0])
        ax_correct_rolling.plot(df[df.target_image=='cheetah'].start_time,
        df[df.target_image=='cheetah'].rewarded.rolling(window).mean(),color=sns.color_palette()[1])
        ax_correct_rolling.plot(df.start_time,df.rewarded.rolling(window).mean(),color=sns.color_palette()[2])
        ax_correct_rolling.set_xlabel('session time (secs)')
        ax_correct_rolling.set_ylabel('% corr ('+str(window)+' trial average)')
        ax_correct_rolling.axhline(0.5,ls='--')
        ax_correct_rolling.set_ylim(0,1)

        # scatter of reaction times 
        sns.scatterplot(data=df, x='start_time',y="reaction_time",hue='target_image',style = 'rewarded', style_order=[True,False],
                    ax=ax_rt,legend=False)

        # histogram of reaction times 
        sns.histplot(data=df, x="reaction_time",hue='target_image',binrange=(0,12),bins=60,
                    ax=ax_rt_hist,legend=False,log_scale=log_hist)

        v = sns.violinplot(data=df, x='rewarded',y="reaction_time",hue='target_image',split = True, 
                    ax=ax_rt_violin)
        v.legend(bbox_to_anchor= (-3.5, -0.2) );
        
        ax_rt_violin.set_yscale('log')
        ax_rt.set_yscale('log')
        if log_hist:ax_rt_hist.set_xscale('log')
        ax_rt_hist.set_xlim(0.1,12)

    fig.text(0.02,0.99, mouse, fontsize=14)
    fig.text(0.08,0.99, date, fontsize=14)
    fig.text(0.02,0.96, 'total percent correct: '+str(df[df.rewarded==True].shape[0]/float(df.shape[0])), fontsize=14)
    fig.text(0.02,0.93, 'mean reaction time: '+str(np.nanmean(df.reaction_time)), fontsize=14)
    # fig.text(0.02,0.90, 'mean rewarded time: '+str(np.nanmean(df[df.rewarded==True].hold_time)), fontsize=14)
    return fig

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

def generate_session_figure(nwb_path):
    io = NWBHDF5IO(nwb_path, mode='r')
    nwb_ = io.read()

    #get task and phase information from the nwb
    if task == 'cheetah_or_elephant':
        pass
    if task == 'bonsai_levertask':
        generate_session(path)
    
    #choose which generate_  function to call
        
    # def generate_session(mouse,date,return_=None,session='combine'):
    #     # paths = glob.glob('/root/work/nwbs_lever/'+mouse+'_'+date+'*_lever.nwb')
    #     paths
    #     if len(paths)> 1:
    #         if session == 'combine':
    #             paths = sort_day_sessions(paths)
    #             df = combine_nwb_sessions(paths)
    #         else:
    #             nwb_path = paths[session]
    #             io = NWBHDF5IO(nwb_path, mode='r')
    #             nwb_ = io.read()
    #             df = nwb_.trials.to_dataframe()
    #     else: 
    #         nwb_path = paths[0]
    #         io = NWBHDF5IO(nwb_path, mode='r')
    #         nwb_ = io.read()
    #         df = nwb_.trials.to_dataframe()


    #     if return_ == 'df':
    #         return df

    #     if return_ == 'fig':

def get_history_sessions(today,tomonth,toyear,number_of_days=10):
    days = []
    mdays = [31,31,28,31,30,31,30,31,31,30,31,30]
    mmonths = [12,1,2,3,4,5,6,7,8,9,10,11]
    for day_minus in range(number_of_days):
        day_ = int(today)-day_minus
        if day_ < 1:
            last_month = int(tomonth)-1
            day_ = mdays[last_month]+day_
            if mmonths[last_month] > int(tomonth):
                lastyear = str(int(toyear) - 1)
                y_m_d = lastyear+'_'+str(mmonths[last_month])+'_'+str(day_)
            else: y_m_d = toyear+'_'+str(mmonths[last_month])+'_'+str(day_)
        else:
            y_m_d = toyear+'_'+tomonth+'_'+str(day_)
        days.extend([y_m_d])
    return days