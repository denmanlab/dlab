import glob,os,filecmp,time,schedule,datetime
from shutil import copytree

def job():
    folders = [r'C:\Users\denma\Desktop\cheetah_or_elephant',
            r'C:\Users\denma\Desktop\bonsai_levertask']
    print('syncing C: to backups at '+datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    for task in folders:
        task_animals = glob.glob(os.path.join(task,'data')+r'\*')
        for folder in task_animals:
            #if an animal doesn't exist within a task data folder on D: or //DENMANLAB/s1, then make it
            try:
                if not os.path.isdir(folder.replace('C:\\Users\denma\Desktop','D:')): os.mkdir(folder.replace('C:\\Users\denma\Desktop','D:'))
            except:pass
            try:
                if not os.path.isdir(folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior/').replace('\\','/')):os.mkdir(folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior/').replace('\\','/'))
            except:pass 

            #this generates a comparison of the sessions within an animal's folder
            C_D = filecmp.dircmp(folder,folder.replace('C:\\Users\denma\Desktop','D:'))
            C_S1 = filecmp.dircmp(folder,folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior/').replace('\\','/'))                        

            #if there are any sessions that are only on the C: drive, then it copies to D: and tries the same to //DENMANLAB/s1 
            if len(C_D.left_only)>0:
                for session in C_D.left_only:
                    print('syncing C: '+session+' to D: at '+datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    copytree(os.path.join(folder,session),os.path.join(folder.replace('C:\\Users\denma\Desktop','D:'),session))
            if len(C_S1.left_only)>0:
                for session in C_S1.left_only:
                    try:
                        print('syncing C '+session+' to DENMANLAB/s1 at '+datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                        copytree(os.path.join(folder,session),folder.replace('C:\\Users\denma\Desktop','//DENMANLAB/s1/behavior').replace('\\','/')+r'/'+session)
                    except:pass
        return
#run at startup
job()

#schedule to recur every day at a certain time.
schedule.every().day.at("01:00").do(job,'It is 01:00')

while True:
    schedule.run_pending()
    time.sleep(60) # wait one minute



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