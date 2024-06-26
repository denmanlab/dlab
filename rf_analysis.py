import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import dlab.psth_and_raster as par
from dlab.generalephys import placeAxesOnGrid, smoothRF
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import scipy.optimize as opt
from skimage.measure import label, regionprops
from skimage.morphology import closing
from tqdm import tqdm
import matplotlib.patches as mpatches

#smooth a 2D image, meant to be space-space of a receptive field
#size = number of pixels to smooth over
def smoothRF(img,size=3):
    """Spatially smooths a computed spatial receptive fields of an already computed receptive field. uses a 2D Gaussian to smooth.

    Parameters
    ----------
    img : np.array
        the 2d spatial receptive field to smooth
    size : int, optional
        the width of the Gaussian used to smooth (default is
        3)

    Returns
    -------
    np.array
        the smoothed input image
    """
    smooth = ndimage.gaussian_filter(img,(size,size))
    return smooth

from scipy.signal import boxcar,convolve
def smooth_boxcar(data,boxcar_size=3):
    """smooths an impulse respone of an already computed receptive field. uses a boxcar to smooth.

    Parameters
    ----------
    data : np.array
        the 1d temporal kernel to smooth
    size : int, optional
        the width of the boxcar used to smooth (default is
        3)

    Returns
    -------
    np.array
        the smoothed input kernel
    """
    smoothed = convolve(data,boxcar(boxcar_size))/boxcar_size
    smoothed = smoothed[boxcar_size/2:len(data)+(boxcar_size/2)]
    return smoothed

def plotsta(sta,taus=(np.linspace(-10,280,30).astype(int)),colorrange=(-0.15,0.15),title='',taulabels=False,nrows=3,cmap=plt.cm.seismic,smooth=None,window = [[0,64],[0,64]]):
    """show the space-space plots of an already computed receptive field for a range of taus.

    Parameters
    ----------
    sta : np.array
        the 1d temporal kernel to smooth
    taus : int, optional
        the width of the boxcar used to smooth (default is
        3)
    colorrange : tuple, optional
        the limits of the colormap. default=(0.35,0.65)
    cmap :string, optional
        the color map to use to plot the receptive field. use any matplotlib colrmap. https://matplotlib.org/stable/gallery/color/colormap_reference.html   default = 'gaussian_2D'
    title : string, optional
        title of the plot. default = ''
    taulabels : bool, optional
            whether or not to label each subplot with the tau. default = False
    nrows : int, optional
        the number of rows of taus to plot. figures out how many columns to plot based on this. default = 3 
    smooth : int, optional
        the width of a Gaussian kernel for spatial smoothing. uses smoothRF function.  default = None
    window : tuple or tuple-like
        the subarea of the receptive field to plot. default = [[0,64],[0,64]]

    Returns
    -------
    np.array
        the smoothed input kernel
    """    
    ncols = np.ceil(len(taus) / nrows ).astype(int)#+1
    fig,ax = plt.subplots(nrows,ncols,figsize=(10,6))
    titleset=False
    m=np.mean(sta[str(taus[3])])
    for i,tau in enumerate(taus):
        axis = ax[int(np.floor(i/ncols))][i%ncols]
        if smooth is None:
            img = sta[str(tau)].T 
        else:
            img = smoothRF(sta[str(tau)].T,smooth)

        axis.imshow(img,cmap=cmap,vmin=colorrange[0],vmax=colorrange[1],interpolation='none')
        axis.set_frame_on(False);
        axis.set_xticklabels('',visible=False);
        axis.set_xticks([]);
        axis.set_yticklabels('',visible=False);
        axis.set_yticks([])
        axis.set_aspect(1.0)
        axis.set_xlim(window[0][0],window[0][1])
        axis.set_ylim(window[1][0],window[1][1])
        if taulabels:
            axis.set_title('tau = '+str(tau),fontsize=8)
        if titleset is not True:
            axis.set_title(title,fontsize=12)
            titleset=True
        else:
            if tau == 0:
                axis.set_title('tau = '+str(tau),fontsize=12)
    plt.tight_layout()
    fig.show()

def twoD_Gaussian(p, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """function for fitting a spatial receptive with a 2-dimensional gaussian.

    Parameters
    ----------
    p : tuple
        the center of the fit
    amplitude : float
        the scaling of the gaussian
    xo : float
        starting x position of fit
    yo : float
        starting y position of fit
    sigma_x : float
        the width of the gaussian in the x dimension
    sigma_y : float
        the width of the gaussian in the y dimension
    theta : float
        the angle between the major and minor axes of the gaussian
    offset : float
        a fixed offset from baseline 

    Returns
    -------
    np.array
        fit 
    """
    x=p[0];y=p[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def fit_rf_2Dgauss(data,center_guess,width_guess=2,height_guess=2):
    """function for fitting a spatial receptive with a 2-dimensional gaussian.

    Parameters
    ----------
    data : tuple
        the data to be fit
    center_guess : tuple, ints
        the center of the fit
    width_guess : int, optional
        the x width position of fit
    height_guess : int, optional 
        the y height  of fit

    Returns
    -------
    popt:  np.array
        fit parameters (popt from opt.curve_fit)
    pcov: np.array
        covariance of parameter (pcov from opt.curve_fit)
    reshaped_to_space: np.array
        the fit reshaped to the dimensions of the input array 
    """
    dataToFit = data.ravel()
    x=np.linspace(0,np.shape(data)[0]-1,np.shape(data)[0])
    y=np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    x, y = np.meshgrid(x, y)
    popt,pcov = opt.curve_fit(twoD_Gaussian,(x,y),dataToFit,p0=(data[center_guess[1]][center_guess[0]], center_guess[1], center_guess[0], width_guess, height_guess, 0, 0))
    reshaped_to_space=(x,y,twoD_Gaussian((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,pcov,reshaped_to_space

def fit_rf_2Dgauss_centerFixed(data,center_guess,width_guess=2,height_guess=2):
    """function for fitting a spatial receptive with a 2-dimensional gaussian. similar to fit_rf_2DGauss, but the position of the Gaussian cannot change with the fit, only the other parameters

    Parameters
    ----------
    data : tuple
        the data to be fit
    center_guess : tuple, ints
        the center of the fit. this will not change during fitting
    width_guess : int, optional
        the x width position of fit
    height_guess : int, optional 
        the y height  of fit

    Returns
    -------
    np.array,np.array,np.array
         fit parameters (popt from opt.curve_fit), covariance of parameter (pcov from opt.curve_fit), the fit reshaped to the dimensions of the input array 
    """
    dataToFit = data.ravel()
    x=np.linspace(0,np.shape(data)[0]-1,np.shape(data)[0])
    y=np.linspace(0,np.shape(data)[1]-1,np.shape(data)[1])
    x, y = np.meshgrid(x, y)
    
    def twoD_Gaussian_fixed(p, amplitude,sigma_x, sigma_y, theta, offset):
        x=p[0];y=p[1]
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-center_guess[1])**2) + 2*b*(x-center_guess[1])*(y-center_guess[0]) 
                                + c*((y-center_guess[0])**2)))
        return g.ravel()
    popt,pcov = opt.curve_fit(twoD_Gaussian_fixed,(x,y),dataToFit,p0=(data[int(center_guess[0])][int(center_guess[1])], width_guess, height_guess, 0, 0))
    reshaped_to_space=(x,y,twoD_Gaussian_fixed((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,pcov,reshaped_to_space

def impulse(sta,center,taus = np.arange(-10,290,10).astype(int)):
    """find the impulse response at a certain position across a range of taus for an already computed receptive field

    Parameters
    ----------
    sta : tuple
        the data to be fit
    center : tuple, ints
        the x,y position to meausure the impulse response
    taus : np.array, optional
        the interval (ints) at which the measure the impulse response. should match the taus of the linear receptive field 

    Returns
    -------
    taus: np.array
        the interval (times) of the impulse response
    impulse: np.array
        the impulse response
    """
    impulse = [sta[str(tau)][int(center[0])][int(center[1])] for tau in taus]
    return (taus,impulse)

#is very, very finicky right now, requires lots of manual tweaking. 
def fitRF(RF,threshold=None,fit_type='gaussian_2D',verbose=False,rfsizeguess=1.2,flipSpace=False,backup_center=None,zoom_int=10,zoom_order=5,centerfixed=False):
    """find the impulse response at a certain position across a range of taus for an already computed receptive field

        Parameters
        ----------
        RF : tuple
            the data to be fit
        threshold : tuple, optional
            the x,y position to meausure the impulse response
        fit_type :string, optional
            the type of fit to try. options are 'gaussian_2D', . default = 'gaussian_2D'
        verbose : bool, optional
            print extra statements during the fit process. default = False
        rfsizeguess : float, optional
            default = 1.2
        flipSpace : bool, optional
            whether or not to transpose the axes. default = False
        backup_center : tuple, optional
            a backup center to use to zoom around if the fit doesn't work. default = None
        zoom_int : int, optional
            the zoom factor of the zoomed in receptive field around the center. default = 10. 
        zoom_order : int, optional
            the order of the filter to smooth a zoomed in receptive field. default = 5
        centerfixed : bool, optional
            whether or not to use a fixed center for the fit. default = False
            
        Returns
        -------
        fit: dictionary
            the averaged spatial RF, 
            the centroid of the fit [max fit]
            a 2D gaussian fit of that spatial RF
            the impulse response at the center of the fit
        """
#   TODO: a fit of that impulse response with: ?? currently not defined.
    if False:#if np.isnan(RF[RF.keys()[0]][0][0]):#check to make sure there is any data in the STRF to try to fit. if not, return the correct data structure filled with None
        fit={};fit['avg_space_fit']=None;fit['params'] = None;fit['cov']=None ;fit['amplitude']=None ;fit['x']=None ;fit['y']=None ;fit['s_x']=None ;fit['s_y']=None ;fit['theta']=None ;fit['offset']=None;fit['center']=None;fit['peakTau']=None;fit['impulse']=None;fit['roughquality']=None
        return fit
    else:

        
        if type(RF)==dict:
            if 'fit' in RF.keys():
                trash = RF.pop('fit') # get rid of the dictionary entry 'fit'; we only want sta tau frames in this dictionary
                taus = [int(i) for i in RF.keys()]
                taus.sort()
                fit={}
            #========================================================================
            #find the taus to average over for the spatial RF
            #first, define the threshold; above this means there is a non-noise pixel somwhere in the space-space
            if threshold == None:
                #set to z sd above mean
                blank = (RF['-10']+RF['0']+RF['10'])/3. # define a blank, noise-only image
                threshold = np.mean(blank)+np.std(blank)*3.
                if verbose:
                    print('threshold: '+str(threshold))
                
            #find the average space-space over only the range of non-noise good 
            avgRF = np.zeros(np.shape(RF[str(int(taus[0]))]))#initialize the average to blank.
            goodTaus = [40,50,60,70,80,90,100]#
            for tau in goodTaus:
                avgRF += RF[str(int(tau))]
            avgRF = avgRF / float(len(goodTaus))
            fit['avg_space']=avgRF

            maximum_deviation = 0;best_center = (0,0)
            for i in np.linspace(24,63,40):
                for j in np.linspace(10,49,40):
                    imp_temp = impulse(RF,(i,j))
                    if np.max(np.abs(imp_temp[1])) > maximum_deviation:
                        best_center = (int(i),int(j))
                        maximum_deviation = np.max(np.abs(imp_temp[1]))
            center = best_center
            imp_temp = impulse(RF,center)
            if verbose:
                print('peak frame tau: '+str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]])))
                print('peak center   : '+str(center))
                print('peak value    : '+str(RF[str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]]))][center[0],center[1]]))
            peak_frame = RF[str(int(imp_temp[0][np.where(np.array(np.abs(imp_temp[1]))==np.max(np.abs(imp_temp[1])))[0][0]]))]
            peak = peak_frame[int(center[0]),int(center[1])]
            #center = (np.where(np.abs(smoothRF(peak_frame,1)) == np.max(np.abs(smoothRF(peak_frame,1))))[0][0],np.where(np.abs(smoothRF(peak_frame,1)) == np.max(np.abs(smoothRF(peak_frame,1))))[1][0])
            
            #========================================================================   
        if type(RF)==np.ndarray:
            fit={}
            fit['avg_space']=RF
            peak_frame = RF
            center = backup_center
            peak = np.max(peak_frame)
        #====fit==================================================================

        if verbose:
            print('peak amp: '+str(peak)+'  threshold: '+str(threshold))
        if np.abs(peak) > threshold * 1.0:
            peak_frame = smoothRF(ndimage.zoom(peak_frame,zoom_int,order=zoom_order),0)
            fit['roughquality']='good'
        else:
            center = backup_center
            imp_temp = impulse(RF,center)
            peak_frame = RF[str(int(100))]
            peak = peak_frame[center[0],center[1]]
            peak_frame = smoothRF(zoom(peak_frame,zoom_int,order=zoom_order),0)
            print('could not find a peak in the RF, using center: '+str(center))
            print('peak amplitude: '+str(peak)+', threshold: '+str(threshold))
            fit['roughquality']='bad'
        fit['center']=center
        fit['center_guess']=center
        fit['fit_image']=peak_frame
        if verbose:
            print('center guess: '+str(center))
        
        #initialize some empty parameters
        fitsuccess=False;retry_fit = False
        best_fit = 10000000#initialize impossibly high
        fit['avg_space_fit']=None;fit['params']=None
        best_fit_output = ((None,None,None,None,None,None,None),600,None)
        try:
            if centerfixed:
                popt,pcov,space_fit = fit_rf_2Dgauss_centerFixed(peak_frame,(center[0]*zoom_int,center[1]*zoom_int),width_guess=rfsizeguess*zoom_int,height_guess=rfsizeguess*zoom_int)
            else:
                popt,pcov,space_fit = fit_rf_2Dgauss(peak_frame,(center[0]*zoom_int,center[1]*zoom_int),width_guess=rfsizeguess*zoom_int,height_guess=rfsizeguess*zoom_int)
        except:
            popt,pcov,space_fit=((None,0,0,0,0,0,0),600,np.zeros((64,64)));print('2D fit failed')
        
        fit['avg_space_fit']=np.array(space_fit)/float(zoom_int)
        fit['params'] = popt    
        fit['cov']=pcov
        fit['amplitude']=popt[0]
        if centerfixed:
            fit['x']=center[1]
            fit['y']=center[0]
            fit['s_x']=popt[1] / float(zoom_int)
            fit['s_y']=popt[2] / float(zoom_int)
            fit['theta']=popt[3]
            fit['offset']=popt[4] / float(zoom_int)
        else:
            fit['x']=popt[1] / float(zoom_int)
            fit['y']=popt[2] / float(zoom_int)
            fit['s_x']=popt[3] / float(zoom_int)
            fit['s_y']=popt[4] / float(zoom_int)
            fit['theta']=popt[5]
            fit['offset']=popt[6] / float(zoom_int)
        #======================================================================== 
    
    #        
        #============get impulse======================================================================== 
        if verbose:
            print('center: '+str(center[0])+' '+str(center[1]))
            
        if fit['avg_space_fit'] is not None:
            center_h = (np.ceil(fit['y']),np.ceil(fit['x']))
            center_r = (np.round(fit['y']),np.round(fit['x']))
            center_l = (np.floor(fit['y']),np.floor(fit['x']))
        try:
            impuls_h = impulse(RF,center_h,taus)[1]
            impuls_r = impulse(RF,center_r,taus)[1]
            impuls_l = impulse(RF,center_l,taus)[1]
            if np.max(np.abs(impuls_h)) > np.max(np.abs(impuls_r)):
                if np.max(np.abs(impuls_h)) > np.max(np.abs(impuls_l)):
                    impuls = impuls_h
                    center = center_h
                else:
                    impuls = impuls_l
                    center= center_l
            else:
                if np.max(np.abs(impuls_r)) > np.max(np.abs(impuls_l)):
                    impuls= impuls_r
                    center= center_r
                else:
                    impuls= impuls_l
                    center= center_l
        except:
            impuls = np.array([0,0,0,0,0,0]) #impuls = np.zeros(len(taus))
            taus = impuls
        if fit_type == 'gaussian_2D':
            #get impulse at the 'center'
            if verbose:
                print('center from fit: '+str(center[0])+' '+str(center[1]))
            #impuls = [RF[str(tau)][center[0]][center[1]] for tau in taus]
            fit['impulse']=(np.array(taus),np.array(impuls))
            peakTau = taus[np.abs(np.array(impuls)).argmax()]
            peakTau = 80
            fit['peakTau']=peakTau
        
        fit['center_usedforfit'] = fit['center']
        fit['center_usedforimp'] = center
        fit['impulse']=(np.array(taus),np.array(impuls))
        #======================================================================== 
        
        return fit 

def show_sta_fit(fit,colorrange=(0.35,0.65),cmap=plt.cm.seismic,title='',contour_levels=3):
    """convenience plotting method for showing spatial and temporal filters pulled from fitting an already computed STRF

        Parameters
        ----------
        fit : dictionary
            the fit data to plot. requires 'impulse' and 'avg_space' keys. 
        colorrange : tuple, optional
            the limits of the colormap. default=(0.35,0.65)
        cmap :string, optional
            the color map to use to plot the receptive field. use any matplotlib colrmap. https://matplotlib.org/stable/gallery/color/colormap_reference.html   default = 'gaussian_2D'
        title : string, optional
            title of the plot. default = ''
        contour_levels : int, optional
            default = 3
        
            
        Returns
        -------
        fig: matplotlib figure handle
            matplotlib figure handle
        """
    if fit is not None:
        fig = plt.figure(figsize=(6,2.75))
        ax_full_space = placeAxesOnGrid(fig,dim=(1,1),xspan=(0,0.32),yspan=(0,0.5))
        ax_zoom_space = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.33,0.63),yspan=(0,0.5))
        ax_zoom_space_filtered = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.56,0.86),yspan=(0,0.5))
        ax_impulse = placeAxesOnGrid(fig,dim=(1,1),xspan=(0.1,0.7),yspan=(0.54,1.0))
        
        ax_impulse.plot(fit['impulse'][0],np.zeros(len(fit['impulse'][0]))+0.5,'r-')
        ax_impulse.plot(fit['impulse'][0],fit['impulse'][1]);
        ax_impulse.set_ylim(colorrange[0],colorrange[1]) 
        ax_impulse.set_xlabel('time (msec)') 
        ax_impulse.set_ylabel('spike probability')
        #ax_impulse.text(1,1,str(np.mean(fit['cov'])))
        
        ax_full_space.plot([fit['y']],[fit['x']],'k+',markersize=6.0)#([fit['center'][0]],[fit['center'][1]],'k+',markersize=6.0)
        ax_full_space.imshow(fit['avg_space'].T,
           cmap=cmap,
           vmin=colorrange[0],vmax=colorrange[1],
            interpolation='none');
        plt.setp(ax_full_space.get_xticklabels(),visible=False);ax_zoom_space_filtered.xaxis.set_major_locator(plt.NullLocator())
        plt.setp(ax_full_space.get_yticklabels(),visible=False);ax_zoom_space_filtered.yaxis.set_major_locator(plt.NullLocator())
        ax_full_space.set_title(title)
        ax_full_space.set_ylim(20,40)
        ax_full_space.set_xlim(12,52)
        ax_full_space.axvline(x=32,linewidth=1,color='r',ls='--')
        
        zoom_size = 5 #number of pixels in each direction from peak center to keep in zooms
        ax_zoom_space.imshow(fit['avg_space'],
                   cmap=cmap,
                   vmin=colorrange[0],vmax=colorrange[1],
                    interpolation='none');
        #ax_zoom_space.plot([fit['x']],[fit['y']],'k+',markersize=6.0)
        ax_zoom_space.set_xlim(fit['center'][1]-zoom_size,fit['center'][1]+zoom_size)
        ax_zoom_space.set_ylim(fit['center'][0]-zoom_size,fit['center'][0]+zoom_size) 
        plt.setp(ax_zoom_space.get_xticklabels(),visible=False);ax_zoom_space_filtered.xaxis.set_major_locator(plt.NullLocator())
        plt.setp(ax_zoom_space.get_yticklabels(),visible=False);ax_zoom_space_filtered.yaxis.set_major_locator(plt.NullLocator())
        

        ax_zoom_space_filtered.imshow(smoothRF(fit['avg_space'],1),
                   cmap=cmap,
                   vmin=colorrange[0]/3.,vmax=colorrange[1]/3.,
                    interpolation='none');
        
        #ax_zoom_space_filtered.plot([fit['x']],[fit['y']],'k+',markersize=6.0)
        ax_zoom_space_filtered.set_xlim(fit['center'][1]-zoom_size,fit['center'][1]+zoom_size)
        ax_zoom_space_filtered.set_ylim(fit['center'][0]-zoom_size,fit['center'][0]+zoom_size)
        plt.setp(ax_zoom_space_filtered.get_xticklabels(),visible=False);ax_zoom_space_filtered.xaxis.set_major_locator(plt.NullLocator())
        plt.setp(ax_zoom_space_filtered.get_yticklabels(),visible=False);ax_zoom_space_filtered.yaxis.set_major_locator(plt.NullLocator())
        
    
        

        
        if 'avg_space_fit' in fit.keys():
            if fit['avg_space_fit'] is not None:
                if np.max(np.abs(fit['impulse'][1])) > np.std(fit['impulse'][1])*2.:
                    ax_full_space.add_patch(Rectangle((fit['y']-zoom_size,fit['x']-zoom_size),zoom_size*2,zoom_size*2,fill=None,ls='dotted'))
                    ax_zoom_space.contour(fit['avg_space_fit'][0],
                                fit['avg_space_fit'][1],
                                fit['avg_space_fit'][2],
                                contour_levels)
                    ax_zoom_space_filtered.contour(fit['avg_space_fit'][0],
                                fit['avg_space_fit'][1],
                                fit['avg_space_fit'][2],
                                contour_levels)
                    ax_full_space.contour(fit['avg_space_fit'][1],
                                fit['avg_space_fit'][0],
                                fit['avg_space_fit'][2],
                                contour_levels)
                    ax_full_space.set_aspect('equal')   

		#                   vmin=-0.08,vmax=0.08);
        #plt.tight_layout()
        return fig
    
#TODO: should be modified to take data of any shape (for example, an LFP trace) to average.
def sta(spiketimes,data,datatimes,taus=(np.linspace(-10,280,30)),exclusion=None,samplingRateInkHz=25):
    """compute a spike-triggered average on three dimensional data. this is typically a movie of the stimulus

        Parameters
        ----------
        spiketimes : np.array
            the times of spikes
        data : tuple-like or np.array
            the stimulus data, with one dimension matching datatimes
        datatimes np.array
            the times of each stimulus in data, should match one of the axes of data
        taus : tuple of np.array, optional
            the taus to calculate. usually provided in milliseconds. default = (np.linspace(-10,280,30))
        exclusion : int, optional
            default = 3
        samplingRateInkHz: float, optional
            multiply the taus by this number.  default=25
        
            
        Returns
        -------
        output: dict
            the average of the input at each tau, where the tau is the key and the computed average is the value. 
        """
    output = {}
    for tau in taus:
        avg = np.zeros(np.shape(data[:,:,0]))
        count = 0
        for spiketime in spiketimes:
            if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.6:
                if exclusion is not None: #check to see if there is a period we are supposed to be ignoring, because of eye closing or otherwise
                    if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.6:
                        index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                        avg += data[:,:,index]
                else:
                    index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                    avg += data[:,:,index]
                count+=1
        output[str(int(tau))]=avg/count
    return output

#should compute a spike-triggered average on n-dimensional data.
#this is typically a movie of the stimulus of shape [frames,x,y], but could be any shape, such as a 1-d noise stimulus or continuous trace.
#as of now, limited to computation of the average along the first axis of 'data'. 
def sta2(spiketimes,data,datatimes,taus=(np.linspace(-10,280,30)),exclusion=None,samplingRateInkHz=30,time_domain=False):
    """#should compute a spike-triggered average on n-dimensional data. this is typically a movie of the stimulus of shape [frames,x,y], but could be any shape, such as a 1-d noise stimulus or continuous trace. as of now, limited to computation of the average along the first axis of 'data'. 

        Parameters
        ----------
        spiketimes : dictionary
            the fit data to plot. requires 'impulse' and 'avg_space' keys. 
        data : tuple, optional
            the limits of the colormap. default=(0.35,0.65)
        datatimes :string, optional
            the color map to use to plot the receptive field. use any matplotlib colrmap. https://matplotlib.org/stable/gallery/color/colormap_reference.html   default = 'gaussian_2D'
        taus : tuple of np.array, optional
            the taus to calculate. usually provided in milliseconds. default = (np.linspace(-10,280,30))
        exclusion : int, optional
            default = 3
        samplingRateInkHz: float, optional
            multiply the taus by this number.  default=30
        time_domain: bool, optional
              default=False

        Returns
        -------
        output: dict
            the average of the input at each tau, where the tau is the key and the computed average is the value. 
        """
    output = {}
    if time_domain:
        taus = taus / 1000.
    else:
        taus = (taus * samplingRateInkHz)/1000.
		
    for tau in taus:
        avg = np.zeros(np.shape(data[0]))
        count = 0
        for spiketime in spiketimes:
            if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                if exclusion is not None: #check to see if there is a period we are supposed to be ignoring, because of eye closing or otherwise
                    if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.5:
                        index = (np.where(datatimes > (spiketime - tau))[0][0]-1) % np.shape(data)[0]
                        avg += data[index]
                else:
                    index = (np.where(datatimes > (spiketime - tau))[0][0]-1) % np.shape(data)[0]
                    avg += data[index]
                count+=1
        output[str(tau)]=avg/count
    return output

class rCorr():
    """
    Class for reverse-correlation analyses such as STA, STC, etc. All analyses take spike times, stimulus times, signal, and tau values. Would like to accommodate functions for 1D and 2D signals. 
    """
    def __init__(self,spike_times,stim_times,signal,taus=np.linspace(-0.01,0.28,30),exclusion=None):
        self.signal      = np.array(signal)
        self.stim_times  = np.array(stim_times)
        self.spike_times = np.array(spike_times)
        self.taus        = np.array(taus)
        
        self.start       = stim_times[0]
        self.end         = stim_times[-1]-0.06
        
        self.stim_spikes = self.spike_times[(self.spike_times >= self.start) & (self.spike_times < self.end)]
        self.spikes_adj  = self.stim_spikes[:,np.newaxis] - self.taus
        
        self.spikes_adj  = self.spikes_adj.T
        
        def find_nearest(array, value):
            idx = (np.abs(array-value)).argmin()
            return idx
        
        if exclusion is not None: 
            # Check if there are time periods we should ignore (eye closing, stim issues, etc.)
            for i in exclusion:
                ex1 = find_nearest(self.stim_spikes,i[0])
                ex2 = find_nearest(self.stim_spikes,i[1])
                self.stim_spikes = np.delete(self.stim_spikes,np.arange(ex1,ex2))
    
    def sta(self):
        output = np.zeros(((len(self.taus),) + self.signal[0].shape))
        output[:] = np.nan
        
        for i,tau in enumerate(self.taus):
            avg = np.zeros(self.signal[0].shape)
            count = 0
            for spike in self.stim_spikes:
                index = (np.where(self.stim_times > (spike - tau))[0][0]-1)
                avg += self.signal[index]
                count+=1
                    
            output[i,:,:] = avg/count
            
        return output,self.taus
        
    def stc(self):
        return 

def plot_sta(sta,taus=np.linspace(-0.01,0.28,30),nrows=3,smooth=None,taulabels=False,**kwargs):
    img = sta
    
    if smooth is not None:
        for i,rf in enumerate(img):
            img[i] = smoothRF(rf,size=smooth)
            
    img = (img - img.mean())/img.std()

    # gmin = img.mean()-(img.std()*3)
    # gmax = img.mean()+(img.std()*3)
    gmin = -4.5
    gmax = 4.5
    
    ncols  = np.ceil(len(taus)/nrows).astype(int)
    
    if 'cmap' in kwargs:
        colormap = kwargs['cmap']
    else: colormap='bwr'
    
    with mpl.rc_context({'xtick.bottom':False,
                          'xtick.labelbottom':False,
                          'ytick.left':False,
                          'ytick.labelleft':False
                          }):
        fig,ax = plt.subplots(nrows,ncols,figsize=(10,6))
        
        for i,tau in enumerate(taus):
            axis = ax[int(np.floor(i/ncols))][i%ncols]
            axis.imshow(np.fliplr(img[i]),vmin=gmin,vmax=gmax,cmap=colormap)
            axis.set_frame_on(False)
            axis.set_aspect(1.0)
            if taulabels==True:
                axis.set_title(f'{int(tau*1000)}ms',fontsize=8,color='k')
            else:
                if tau==0:
                    axis.set_title('0ms',fontsize=8,color='k')
                
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'] +f' min={img.min():.3f} max={img.max():.3f}',
                     fontsize=10,color='k',y=0.85,fontweight='semibold')
        
    plt.subplots_adjust(wspace=0.1,hspace=-0.6)
    
    if 'facecolor' in kwargs:
        fig.set_facecolor(kwargs['facecolor'])
        

def sta_with_subfields(spiketimes,data,datatimes,taus=(np.linspace(-10,280,30)),exclusion=None,samplingRateInkHz=25):
    """compute a spike-triggered average on three dimensional data. this is typically a movie of the stimulus, similar to sta(). the difference is that this calculates bright and dark separately as well.

        Parameters
        ----------
        spiketimes : np.array
            the times of spikes
        data : tuple-like or np.array
            the stimulus data, with one dimension matching datatimes
        datatimes np.array
            the times of each stimulus in data, should match one of the axes of data
        taus : tuple of np.array, optional
            the taus to calculate. usually provided in milliseconds. default = (np.linspace(-10,280,30))
        exclusion : int, optional
            default = 3
        samplingRateInkHz: float, optional
            multiply the taus by this number.  default=25
        
            
        Returns
        -------
        output: dict
            the average of the input at each tau, where the tau is the key and the computed average is the value. 
        output:_ON dict
            the average of the input bright pixels of the input only at each tau, where the tau is the key and the computed average is the value. 
        output_OFF: dict
            the average of the input dark pizxels of the input only at each tau, where the tau is the key and the computed average is the value. 
        """    
    output = {}
    output_ON = {}
    output_OFF = {}
    for tau in taus:
        avg = np.zeros(np.shape(data[:,:,0]))
        avg_ON = np.zeros(np.shape(data[:,:,0]))
        avg_OFF = np.zeros(np.shape(data[:,:,0]))
        count = 0
        for spiketime in spiketimes:
            if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.6:
                if exclusion is not None: #check to see if there is a period we are supposed to be ignoring, because of eye closing or otherwise
                    if spiketime > datatimes[0] and spiketime < datatimes[-1]-0.6:
                        index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                        avg += data[:,:,index]
                        avg_ON += (data[:,:,index] + 1 ) / 2
                        avg_OFF += (data[:,:,index]- 1 ) / 2
                else:
                    index = (np.where(datatimes > (spiketime - tau*samplingRateInkHz))[0][0]-1) % np.shape(data)[2]
                    avg += data[:,:,index];avg_ON += (data[:,:,index] + 1 ) / 2;avg_OFF += (data[:,:,index] - 1 ) / 2
                count+=1
        output[str(int(tau))]=avg/count
        output_ON[str(int(tau))]=avg_ON/count# - 0.5
        output_OFF[str(int(tau))]=avg_OFF/count# + 0.5
    return output,output_ON,output_OFF

#from https://github.com/JesseLivezey/gabor_fit/blob/master/gabor_fit/fit.py
def generate_gabor(p, pixels_x,pixels_y, theta, stdx, stdy, lamb, phase):
    """
    Generate a gabor filter based on given parameters.
    Parameters
    ----------
    pixels : tuple of ints
        Height and width of patch.
    p : (x,y)
        x : float
            x Location of center of gabor in pixels.
        y : float
            y Location of center of gabor in pixels.
    pixels_x: float
        number of pixels in x dimension of the fit
    pixels_y: float
        number of pixels in y dimension of the fit
    theta : float
        Rotation of gabor in plane in degrees.
    stdx : float
        Width of gaussian window along rot(x) in pixels.
    stdy : float
        Width of gaussian window along rot(y) in pixels.
    lamb : float
        Wavelength of sine funtion in pixels along rot(x).
    phase : float
        Phase of sine function in degrees.
    Returns
    -------
    gabor : ndarray
        normalized fit parameters
    """
    #print (pixels_x,pixels_y)
    x=p[0];y=p[1]
    x_coords = np.arange(0, int(pixels_x))#pixels[0])
    y_coords = np.arange(0, int(pixels_y))#pixels[1])
    xx, yy = np.meshgrid(x_coords, y_coords)
    unit2rad = 2. * np.pi 
    deg2rad = 2. * np.pi / 360.
    xp = (xx - x) * np.cos(deg2rad * theta) - (yy - y) * np.sin(deg2rad * theta)
    yp = (xx - x) * np.sin(deg2rad * theta) - (yy - y) * np.cos(deg2rad * theta)
    gabor = (np.exp(-xp**2 / (2. * stdx**2) - yp**2 / (2. * stdy**2)) *
             np.sin(unit2rad * (xp / lamb) - deg2rad * phase))
    norm = np.sqrt((gabor**2).sum())
    g = gabor/norm
    return g.ravel()

def fitgabor_2(data,pixels=(10,10),x=32.,y=32.,theta=0.,stdx=3.,stdy=3.,lamb=1.5,phase=0.):
    """function for fitting a spatial receptive with a 2-dimensional Gabor.

    
    Parameters
    ----------
    data : tuple
        the data to be fit
    pixels : tuple, ints
        the number of pixels in the fit
    x: float
        starting x position of fit
    y : float
        starting y position of fit
    height_guess : int, optional 
        the y height  of fit
    theta : float
        the angle between the major and minor axes of the gaussian
    stdx : float
        the width of the gaussian in the x dimension
    stdy : float
        the width of the gaussian in the y dimension
    lamb : float
        Wavelength of sine funtion in pixels along rot(x).
    phase : float
        Phase of sine function in degrees.

    Returns
    -------
    popt:  np.array
        fit parameters (popt from opt.curve_fit)
    pcov: np.array
        covariance of parameter (pcov from opt.curve_fit)
    reshaped_to_space: np.array
        the fit reshaped to the dimensions of the input array 
    """
    popt,pcov = opt.curve_fit(generate_gabor,(x,y),data.ravel(),p0=(int(pixels[0]),int(pixels[1]),theta,stdx,stdy,lamb,phase))
    reshaped_to_space=(x,y,generate_gabor((x,y),*popt).reshape(np.shape(data)[1],np.shape(data)[0]))
    return popt,reshaped_to_space,pcov

def check_rfs_in_df(df_rf,sds=4):
    """this function will check if the RFs computed and stored in this dataframe have signal in them, adding three rows to the input df:
          rf_computed: if any RF STA was calculated at all
          good: if there was a non-noise RF
          rf_color: the color of the noise used to generate the "good" RF, if there was a good one

    
    Parameters
    ----------
    df_rf:  pandas.DataFrame
        a pandas DataFrame with some receptive fields computed from uv and green noise to check. must have a 'g_avg_space' column, and 'g_fit_image' and 'u_fit_image' columns
    sds : int, optional
        the number of SDs a pixels has to be above in order to be considered a real RF pixel

    Returns
    -------
    df_rf:  pandas.DataFrame
        the input DataFrame 
    """
    x=sds
    rf_computed = []
    good_rf = []
    rf_color = []
    for ind in df_rf.index:
        if type(df_rf.g_avg_space[ind]) != str:
            # measure some things about the computed rf
            rf_computed.extend([True])
            
            gr = df_rf.g_fit_image[ind]
            im = smoothRF(gr,.5);
            thresh = np.mean(im)+x*np.std(im)
            im[np.abs(smoothRF(gr,.6))<thresh]=0
            gr_pixels = np.nansum(np.abs(im))
            
            uv = df_rf.u_fit_image[ind]
            im = smoothRF(uv,.5);
            thresh = np.mean(im)+x*np.std(im)
            im[np.abs(smoothRF(uv,.6))<thresh]=0
            uv_pixels = np.nansum(np.abs(im))
    
            if gr_pixels>0 or uv_pixels>0:
                good_rf.extend([True])
                if uv_pixels>gr_pixels:
                    im = smoothRF(uv,.5)
                    rf_color.extend(['uv'])
                else:
                    im = smoothRF(gr,.5)
                    rf_color.extend(['green'])
            else: 
                rf_color.extend([np.nan]);good_rf.extend([False])
        else:
            rf_computed.extend([False])
            good_rf.extend([False])
            rf_color.extend([np.nan])
    df_rf['rf_computed']=rf_computed
    df_rf['good_rf']=good_rf
    df_rf['rf_color']=rf_color
    
    return df_rf

def show_impulse(a,center):
    """this function compute an impulse response from a receptive field and plot it. both the impulse and the spatial receptive field at 80 msec tau are plotted.  

    
    Parameters
    ----------
    a:  dict
        a dictionary containing a computed receptive field, where the keys of the dict are the tau and the values are the spatial RF at that tau
    center : tuple, ints
        the pixels in each receptive field at which to measure the impulse response

    Returns
    -------
    None
    """

    center=(41,32)
    i = impulse(a,center,taus=np.linspace(-10,280,30).astype(int))
    plt.plot(i[0],i[1])
    plt.ylim(108,148)
    plt.gca().axhline(128,ls='--')
    plt.figure()
    plt.imshow(a['80'])
    plt.gca().scatter(center[1],center[0],color='w',alpha=0.3)

def segRF(array, flip=False, kernel=1,colormap='PiYG'):
    '''
    Highlights RF using thresholding and image segmentation. Requires manual selection of the index (tau value) of interest in the series of RFs
    '''
    im = array
    img_gauss = smoothRF(im,kernel)

    bn_img = np.zeros([img_gauss.shape[0],img_gauss.shape[1]])
    bn_img = abs(img_gauss) >= (abs(img_gauss).max()-np.std(img_gauss))

    sklabels = label(closing(bn_img))

    sizes = []
    for region in regionprops(sklabels):
        sizes.append(region.area)

    max_roi = np.asarray(sizes).argmax()


    centroid = regionprops(sklabels)[max_roi].centroid
    centroid = (centroid[1],centroid[0])
    major_length = regionprops(sklabels)[max_roi].major_axis_length
    minor_length = regionprops(sklabels)[max_roi].minor_axis_length
    eccentricity = regionprops(sklabels)[max_roi].eccentricity
    orientation = regionprops(sklabels)[max_roi].orientation*180/np.pi
    if flip == True:
        orientation = -orientation
    perimeter = regionprops(sklabels)[max_roi].perimeter
    
    data = {'centroid':centroid,
            'major_length':major_length,
            'minor_length':minor_length,
            'eccentricity': eccentricity,
            'orientation': orientation,
            'perimeter':perimeter
           }

    ellipse = mpatches.Ellipse(centroid,minor_length,major_length,orientation,
                              fill=False, edgecolor='r', linewidth=2)
    
    fig,ax = plt.subplots()
    ax.imshow(img_gauss,cmap=colormap,vmin=img_gauss.min(),
            vmax=(abs(img_gauss).max()-np.std(img_gauss)))
    ax.add_patch(ellipse)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return fig, data

def sweep_rf(spiketimes, stim_data, pre=0, post=7,binsize=0.05):
    Nbins = int((post+pre)/binsize)-1
    features = np.unique(stim_data.ori)
    features.sort()
    Ntrials = int(stim_data.shape[0]/len(features))

    psth_all=np.zeros((len(features),Nbins))
    bytrial_all=np.zeros((len(features),Ntrials,Nbins))
    var_all = np.zeros((len(features),Nbins))

    for i,feat in enumerate(features):
        stmtimes = stim_data.start_time[stim_data.ori == feat].values
        psth_,bytrial,var = par.psth_arr(spiketimes,stmtimes,pre=pre,post=post,binsize=binsize,variance=True)
        psth_all[i,:] = psth_
        bytrial_all[i,:,:] = bytrial
        var_all[i,:] = var
    
    return(psth_all,bytrial_all,var_all)

def sweep_proj(data,directions,method):
    '''
    Generates back-projection map of receptive field estimation based on sweep psth data

    DATA: directions x mean psth
        NOT by trial
        Must be numpy array
    DIRECTIONS: 1D list or array of angles of presentation
    METHOD:
        0 = Arithmetic
        1 = Geometric
    '''
    
    sz =  data.shape[1]

    pad = int(np.ceil(np.sqrt(2)*sz-sz)/2)
    pad_dat = np.zeros(int(sz+2*pad))
    pad_dat[:] = np.nan

    cj = np.asmatrix(np.tile(np.arange(-(sz)/2,(sz)/2,1),(sz,1)))
    ci=cj.H

    map_ = np.ones((sz,sz))
    map_[:,:] = method

    for i in range(len(directions)):
        t = directions[i]*np.pi/180
        pad_dat[pad:pad+sz] = data[i]
        rcj = np.asmatrix(np.round(cj*np.cos(t)-ci*np.sin(t)+np.ceil(pad+sz/2)),dtype='int16')
        dat_array = pad_dat[rcj] 

        if method != 0:
            map_ = map_*dat_array
            map_ = map_/(i+1)
        else:
            map_ = map_ + dat_array

    return map_

def mask_grating(spiketimes, stim_data, xpos, ypos, pre, post, binsize):
    '''
    This function is dependent on column titles I created for my NWB files. The column of x coordinates is named "PosX" and the column of Y coordinates is named "PosY". Start time for the stimulus is named 'start_time'

    INPUTS
    spiketimes: 1D array of spike times for a single unit
    stim_data: A pandas dataframe of experiment data requiring specific column names. See above
    xpos, ypos: list of *unique* x and y positions for stimulus presentation. 
    pre, post: time before and after stim presentation for raster/psth
    binsize: desired binsize for analysis

    OUTPUTS
    labels: list of xy coordinate combinations in order of iteration
    psth_all: mean psth for the unit at a given xy coordinate
    bytrial_all: psth for each trial at a given xy 
    var_all: variance (relative to mean psth) for each unit at a given xy

    Juan Santiago 12/07/2022
    '''
    if 'start_time' and 'PosX' and 'PosY' not in stim_data.columns:
        raise Exception('stim_data requires columns named "start_time", "PosX", and "PosY"')

    Nbins = int((pre+post)/binsize-1)

    Ntrial = int(stim_data.shape[0]/(len(ypos)*len(xpos)))

    psth_all    = np.zeros(((len(ypos)*len(xpos),Nbins)))
    var_all     = np.zeros(((len(ypos)*len(xpos),Nbins)))
    bytrial_all = np.zeros((len(ypos)*len(xpos),Ntrial,Nbins))
    labels      = []

    k=0
    for i, y in enumerate(ypos):
        for j, x in enumerate(xpos):
            # Dependent on column titles
            stim_times = stim_data.start_time[stim_data.PosY == y][stim_data.PosX == x].values
            psth_,bytrial_,var_ = par.psth_arr(spiketimes,stim_times,pre=pre,post=post,binsize = binsize, variance=True)
            psth_all[k,:] = psth_
            var_all[k,:]  = var_
            bytrial_all[k,:,:] = bytrial_
            labels.append((x,y))
            k += 1

    return labels, psth_all, bytrial_all, var_all
