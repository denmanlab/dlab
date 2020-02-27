# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:39:18 2014

@author: danieljdenman
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import cv2
import os

def loadEyeTraces(filepath):
    eyedata={}
    f = h5py.File(filepath,'r')
    for i,key in enumerate(f.keys()):
        if key in ['led_positions','pupil_positions','pupil_area']:
            eyedata[key] = np.array(f[key]).astype(np.int16)
    f.close()
    #f = filepath.replace('h5')
    #timestamps from text files
    return eyedata
    
def plotEyePosition(eyedata,mn=-80,mx=10):
    plt.figure()
    bins = 20
    x = eyedata['pupil_positions'][:,0] - eyedata['led_positions'][:,0]
    y = eyedata['pupil_positions'][:,1] - eyedata['led_positions'][:,1]
    plt.plot(x,y,'k-')
    plt.xlim(mn,mx);plt.ylim(mn,mx)
    plt.figure()
    plt.plot(np.linspace(x.min(),x.max(),num=bins),np.histogram(x,bins)[0])
    plt.figure()
    plt.plot(np.linspace(y.min(),y.max(),num=bins),np.histogram(y,bins)[0])
    plt.figure()
    plt.imshow(np.histogram2d(x,y,bins)[0],cmap=plt.cm.hot,interpolation='none')
    
def smoothLED(eyedata):
    eyedata['led_positions'][:,1] = ndimage.filters.gaussian_filter1d(eyedata['led_positions'][:,1],30)
    eyedata['led_positions'][:,0] = ndimage.filters.gaussian_filter1d(eyedata['led_positions'][:,0],30)
    
def smoothPupil(eyedata):
    eyedata['pupil_positions'][:,1] = ndimage.filters.gaussian_filter1d(eyedata['pupil_positions'][:,1],10)
    eyedata['pupil_positions'][:,0] = ndimage.filters.gaussian_filter1d(eyedata['pupil_positions'][:,0],10)
    
def trigger(eyedata,triggers,window,**kwargs):
    eyeOut={}
    eyeOut['pupil_positions']=[]
    eyeOut['led_positions']=[]
    eyeOut['pupil_area']=[]
    if 'combine' in kwargs.keys():
        f = plt.gcf()
        plt.figure(f.number)
    for trig in triggers:
        
        eyeOut['pupil_positions'].append(eyedata['pupil_positions'][trig:trig+window,:])
        eyeOut['led_positions'].append(eyedata['led_positions'][trig:trig+window,:])
        eyeOut['pupil_area'].append(eyedata['pupil_area'][trig:trig+window])
        
        #print trig
        if 'plot' in kwargs.keys() and kwargs['plot']=='position':
            if 'color' in kwargs.keys():
                color = ''.join((kwargs['color'],'-o'))
            else:
                color =''.join(('k','-o'))
            traceX = np.array(eyedata['pupil_positions'][trig:trig+window,0]-eyedata['led_positions'][trig:trig+window,0])
            traceY = np.array(eyedata['pupil_positions'][trig:trig+window,1]-eyedata['led_positions'][trig:trig+window,1])            
            plt.plot(traceX-traceX[0],
                     traceY-traceY[0],
                        color)
            plt.ylim(-30,30);plt.xlim(-30,30)
        
        if 'plot' in kwargs.keys() and kwargs['plot']=='area':
            if 'color' in kwargs.keys():
                color = kwargs['color'] 
            else:
                color = 'k'
            plt.plot(range(window),eyedata['pupil_area'][trigger:trigger+window],''.join((color,'-o')));plt.ylim(-100,-10);plt.xlim(-100,-10)
    
    return eyeOut
#movie:    
def triggerMovie(movie,triggers,numberofframes,name='output',**kwargs):          
    
    allframes = [np.linspace(start,start+numberofframes-1,numberofframes) for start in triggers] 
    framestowrite = np.concatenate(allframes).astype(int)
    currentframe = 0
    
    # Define the codec and create the VideoCapture and VideoWriter objects
    cam=cv2.VideoCapture(movie) 
    size = (int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
           int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    print size
    fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    out = cv2.VideoWriter(os.path.join(os.path.dirname(movie),name+'.avi'),fourcc, 15.0, size)
    
    #read, display and writing only the frames defined by triggers and number of frames 
    while True:
        ret, img = cam.read()                      
        if (type(img) == type(None)):
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if currentframe in framestowrite:
            out.write(img)
            cv2.imshow('frame', gray)
        
        #print ret
        currentframe+=1
        if (0xFF == ord('q') & cv2.waitKey(1)) or img.size == 0:
            break
    cam.release()
    out.release()   
    cv2.destroyAllWindows()
    
def triggerMovieAvg(movie,triggers,numberofframes,name='output',**kwargs):          
    
    allframes = [np.linspace(start,start+numberofframes-1,numberofframes) for start in triggers] 
    framestowrite = np.concatenate(allframes).astype(int)
    currentframe = 0

    
    # Define the codec and create the VideoCapture and VideoWriter objects
    cam=cv2.VideoCapture(movie) 
    size = (int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
           int(cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    print size
    fourcc = cv2.cv.CV_FOURCC(*'DIVX')
    out = cv2.VideoWriter(os.path.join(os.path.dirname(movie),name+'_trialaverage.avi'),fourcc, 15.0, size)
    
    avg = [np.zeros((size[0],size[1],3)) for i in range(numberofframes)]
    frameno = 0
    #read, display only the frames defined by triggers and number of frames 
    while True:
        ret, img = cam.read()                      
        if (type(img) == type(None)):
            break
        
        if currentframe in framestowrite :
            #out.write(img)
            cv2.imshow('frame', img)
            avg[frameno]+=img
            frameno+=1
        else:
            frameno = 0
            
        #print ret
        currentframe+=1
        if (0xFF == ord('q') & cv2.waitKey(1)) or img.size == 0:
            break
        
    cam.release()
    
    averagemovie = [img / len(triggers) for img in avg] # average
    
    #writing only the frames defined by triggers and number of frames 
    for img in averagemovie:
        out.write(img.astype('u1'))
    out.release()   
    
    cv2.destroyAllWindows()
        
#    startFrame = triggers[0]
#    for frame in numFramesInMovie:
#        if frame in np.linspace(startFrame,startFrame+numberofframes-1,numberofframes).astype(int):
#            #start writing 
#
#            start           
            
#plt.plot(o0['mean'][0]-o0['mean'][0][0],o0['mean'][1]-o0['mean'][1][0])
#plt.plot(o45['mean'][0]-o45['mean'][0][0],o45['mean'][1]-o45['mean'][1][0])
#plt.plot(o90['mean'][0]-o90['mean'][0][0],o90['mean'][1]-o90['mean'][1][0])
#plt.plot(o135['mean'][0]-o135['mean'][0][0],o135['mean'][1]-o135['mean'][1][0])
    
