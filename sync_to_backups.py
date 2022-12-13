import glob,os,filecmp,time,schedule,datetime, sys, getopt, requests
from shutil import copytree
# msvcrt is a windows specific native module
import msvcrt
import time
from dlab import dlabbehavior as db
import pandas as pd

from slack import WebClient
sc = WebClient(token=os.environ['SLACK_BEHAVIOR_BOT_TOKEN'])



# asks whether a key has been acquired
def kbfunc():
    #this is boolean for whether the keyboard has bene hit
    x = msvcrt.kbhit()
    if x:
        #getch acquires the character encoded in binary ASCII
        ret = msvcrt.getch()
    else:
        ret = False
    return ret


def job():
    folders = [r'C:\Users\denma\Desktop\cheetah_or_elephant',
            r'C:\Users\denma\Desktop\bonsai_levertask']
    print('syncing C: to backups at '+datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    for task in folders:
        task_animals = glob.glob(os.path.join(task,'data')+r'\*')
        for folder in task_animals:
            make_plots=False #used later to determine if we make plots for this animal today
            #if an animal doesn't exist within a task data folder on E: or //DENMANLAB/s1, then make it
            try:
                if not os.path.isdir(folder.replace('C:\\Users\denma\Desktop','E:')): os.mkdir(folder.replace('C:\\Users\denma\Desktop','E:'))
            except:pass
            try:
                if not os.path.isdir(folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior/').replace('\\','/')):os.mkdir(folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior/').replace('\\','/'))
            except:pass 

            #this generates a comparison of the sessions within an animal's folder
            C_D = filecmp.dircmp(folder,folder.replace('C:\\Users\denma\Desktop','D:'))
            C_E = filecmp.dircmp(folder,folder.replace('C:\\Users\denma\Desktop','E:'))
            C_S1 = filecmp.dircmp(folder,folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior/').replace('\\','/'))                        

            #if there are any sessions that are only on the C: drive, then it copies to E: and tries the same to //DENMANLAB/s1 
            if len(C_D.left_only)>0:
                for session in C_D.left_only:
                    if len(C_E.left_only)>0:
                        if session in C_E.left_only:
                            print('syncing C: '+session+' to E: at '+datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                            #makenwb
                            db.make_lever_NWB(os.path.join(folder,session)) 
                            make_plots=True
                            #copy it all, including the nwb we just made
                            copytree(os.path.join(folder,session),os.path.join(folder.replace('C:\\Users\denma\Desktop','E:'),session))
            if len(C_S1.left_only)>0:
                for session in C_S1.left_only:
                    try:
                        print('syncing C '+session+' to DENMANLAB/s1 at '+datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                        copytree(os.path.join(folder,session),folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior').replace('\\','/')+r'/'+session)
                    except:pass
           
            #make today's figures
            if make_plots:
                animal = os.path.basename(folder)
                datestring=datetime.datetime.now().strftime("%Y_%m_%d").replace("_0","_")
                if task == folders[0]:#cheetah_or_elephant
                    pass # fr nw
                if task == folders[1]:#lever
                    #generate figures
                    fig,ax = db.generate_session_lever(animal,datestring,session='combine',return_='fig')
                    df = db.generate_session_lever(animal,datestring,session='combine',return_='df')
                    print(df)
                    date_folder = glob.glob(os.path.join(folder,animal+'_'+datestring+'*'))[-1] 
                    if not os.path.exists(os.path.join(date_folder,'figs')):os.makedirs(os.path.join(date_folder,'figs'))
                    fig.savefig(os.path.join(date_folder,'figs','summary.png'))
                    fig.savefig(os.path.join(date_folder,'figs','summary.eps'),format='eps')

                    #go over previous sessions and make across-session figures 
                    #first, get the strings for the previous ten days. do gymnastics to formats date for bonsai task outputs. 
                    # today = datestring.split('_')[2]
                    # tomonth =  datestring.split('_')[1]
                    # toyear = datestring.split('_')[0]
                    # days = []
                    # mdays = [31,31,28,31,30,31,30,31,31,30,31,30]
                    # mmonths = [12,1,2,3,4,5,6,7,8,9,10,11]
                    # for day_minus in range(10):
                    #     day_ = int(today)-day_minus
                    #     if day_ < 1:
                    #         last_month = int(tomonth)-1
                    #         day_ = mdays[last_month]+day_
                    #         if mmonths[last_month] > int(tomonth):
                    #             lastyear = str(int(toyear) - 1)
                    #             y_m_d = lastyear+'_'+str(mmonths[last_month])+'_'+str(day_)
                    #         else: y_m_d = toyear+'_'+str(mmonths[last_month])+'_'+str(day_)
                    #     else:
                    #         y_m_d = toyear+'_'+tomonth+'_'+str(day_)
                    #     days.extend([y_m_d])
        
                    # dfs = [db.generate_session(folder,datestring,session='combine',return_='df') ]
                    # df = pd.concat(dfs,ignore_index=True)
                    # fig = db.across_session_plots_lever(df)
                    # if not os.path.exists(os.path.join(folder,'figs')):os.makedirs(os.path.join(folder,'figs'))
                    # fig.savefig(os.path.join(folder,'figs','recent_sessions.png'))
                    # fig.savefig(os.path.join(folder,'figs','recent_sessions.eps'),format='eps')

                    #send figures to slack     
                    # message = 'behavior updates for '+animal+' on '+datestring
                    sc.chat_postMessage( channel="#behavior_plot_bot", text=message)
                    response = sc.files_upload(channels="#behavior_plot_bot",
                        file=os.path.join(date_folder,'figs','summary.png'),
                        title='summary '+animal+' on '+datestring)
                    # response = sc.files_upload(channels="#behavior_plot_bot",
                    #     file=os.path.join(folder,'figs','recent_sessions.png'),
                    #     title='summary '+folder+' on '+datestring)

    print('done syncing C: to backups at '+datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))    
       

    return
#run at startup
print('backing up C to DENMANLAB')
job()

#schedule to recur every day at a certain time.
schedule.every().day.at("01:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute
    #if we got a keyboard hit
    x = kbfunc()
    if x != False and x.decode() == 'u':
        #we got the key!
        #because x is a binary, we need to decode to string
        #use the decode() which is part of the binary object
        #by default, decodes via utf8
        #concatenation auto adds a space in between
        job()


#---------------------EXTRA STUFF FROM DEV------------------------------------------------------

def copytree2(source,dest):
    os.mkdir(dest)
    dest_dir = os.path.join(dest,os.path.basename(source))
    copytree(source,dest_dir)


# folders = [r'C:\Users\denma\Desktop\cheetah_or_elephant',
#           r'C:\Users\denma\Desktop\bonsai_levertask']


# for folder in folders:
#     C_D = filecmp.dircmp(os.path.join(folder,'data'),os.path.join('D:\\',os.path.basename(folder),'data'))
#     # C_S1 = filecmp.dircmp(os.path.join(folder,'data'),os.path.join('DENMANLAB','s1','behavior',os.path.basename(folder),'data'))
#     if len(C_D.left_only)>0:
#         for mouse in C_D.left_only:
#             copytree2(os.path.join(folder,'data',mouse),os.path.join('D:\\',os.path.basename(folder),'data',mouse))


# for sess in C_D.left_only:
#     copytree2(folder+r'/'+sess,os.path.join('D:\\','cheetah_or_elephant','data','c63',sess))