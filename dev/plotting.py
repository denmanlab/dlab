import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.ndimage as ndimage

import scipy.optimize as opt
from skimage.measure import label, regionprops
from skimage.morphology import closing
import matplotlib.patches as mpatches

import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd


#smooth a 2D image, meant to be space-space of a receptive field
#size = number of pixels to smooth over
def smoothRF(img,size=3):
    smooth = ndimage.gaussian_filter(img,(size,size))
    return smooth

def plot_sta(sta,taus=np.linspace(-0.01,0.28,30),nrows=3,smooth=None,taulabels=False,**kwargs):
    img = sta
    
    if smooth is not None:
        for i,rf in enumerate(img):
            img[i] = smoothRF(rf,size=smooth)
            
    img = (img - img.mean())/img.std()

    # gmin = img.mean()-(img.std()*3)
    # gmax = img.mean()+(img.std()*3)
    gmin = -4
    gmax = 4
    
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
            axis.imshow(img[i],vmin=gmin,vmax=gmax,cmap=colormap)
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
        
# From Github https://github.com/michi-d/receptive_field_reverse_correlation
    
def lowpass1_array(signal,tau):
    '''First order lowpass filter'''
    N     = len(signal)
    tau   = float(tau)
    out   = np.zeros(signal.shape)
    alpha = tau/(tau+1)
    
    out[0,:] = signal[0,:] #initial condition
    for i in range(1,N):
        out[i,:] = signal[i,:] * (1. - alpha) + out[i-1,:]* alpha
        
    return out
    
def rf_centroid(sta, taus = np.linspace(-.01,.28,30), colormap = 'bwr',smooth=None):
    if smooth is not None:
        R_smooth = sta.copy()
        for i,rf in enumerate(sta):
            R_smooth[i] = smoothRF(rf,size=smooth)
            
    
    R = (sta - sta.mean())/sta.std()
    
    for i in range(len(R)):
       R[i]        = np.fliplr(R[i])
       R_smooth[i] = np.fliplr(R_smooth[i])

    minimum  = np.abs(R_smooth).argmax()
    t,i,j    = np.unravel_index(minimum, R.shape)
    maximum   = R_smooth[t,i,j]
    
    max_i,max_j,max_t = i,j,t

    def func(x, a, x0, sigma): # gauss distribution
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    def func_gaussian(x, a, x0, sigma): # gauss distribution
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    # fit gauss in i-direction (elevation)
    ydata = R[int(round(max_t)),:,int(round(max_j))]
    ydata = ydata - ydata.mean()

    xdata = np.arange(len(ydata))
    try:
        popt, pcov = curve_fit(func, xdata, ydata, p0 = (maximum, max_i, smooth), maxfev = 2000)
        centers_i  = popt[1]
        a_i        = popt[0]
        sigma_i    = popt[2]
    except:
        return

    # fit gauss in j-direction (azimuth)
    ydata = R[int(round(max_t)),int(round(max_i)),:]
    ydata = ydata - ydata.mean()
    xdata = np.arange(len(ydata))
    try:
        popt, pcov = curve_fit(func, xdata, ydata, p0 = (maximum, max_j, smooth), maxfev = 2000)
        centers_j  = popt[1]
        a_j        = popt[0]
        sigma_j    = popt[2]
    except:
        return
    
    # fit gauss in t-direction
    ydata = R[:,int(round(max_i)),int(round(max_j))]
    ydata = ydata - ydata.mean()
    xdata = np.arange(len(ydata))
    try:
        popt, pcov = curve_fit(func, xdata, ydata, p0 = (maximum, max_t, smooth), maxfev = 2000)
        centers_t  = popt[1]
        a_t        = popt[0]
        sigma_t    = popt[2]
    except:
        return
    
    with sns.axes_style("white"), sns.axes_style("ticks"):
        fig = plt.figure(figsize = (10,5))
        gs = gridspec.GridSpec(2,2)
        ax = plt.subplot(gs[0:2,0])
        
    #Plot RF with Center Marked
    plt.imshow(R_smooth[int(round(max_t)),:,:],
               cmap = colormap,
               vmin = -np.abs(maximum),
               vmax = np.abs(maximum),
               origin='lower')
    ax.axhline(centers_i, color = 'k', linewidth = 1, linestyle = '--') #Max along elevation
    ax.axvline(centers_j, color = 'k', linewidth = 1, linestyle = '--') #Max along azimuth

    # azimuth distribution
    ax.plot(np.arange(R.shape[2]),
            R[int(round(max_t)),int(round(max_i)),:],
            color = 'r'
           )
    ax.plot(np.arange(0, R.shape[2]),
            func_gaussian(np.arange(0, R.shape[2]), a_j, centers_j,sigma_j),
            color = 'k'
           )

    # Elevation Distribution
    ax.plot(R[int(round(max_t)),:,int(round(max_j))],
            np.arange(R.shape[1]), 
            color = 'r'
           )
    ax.plot(func_gaussian(np.arange(0, R.shape[1]), a_i, centers_i,sigma_i),
            np.arange(0, R.shape[1]), 
            color = 'k'
           )

    # Mark phi and Z (Correlation strength)
    phi = (R.shape[0] - 1 - centers_i)*(125./R.shape[0]) + 125./R.shape[0]/2.
    # z   = -1 * ((R.shape[1] - 1 - centers_j)*(95./R.shape[1]) + 95./R.shape[1]/2. - 47)
    t_gauss = func_gaussian(np.arange(0, R.shape[0]), a_t, centers_t,sigma_t)
    z  = max(t_gauss, key=abs)

    ax.text(0.65,0.85, 'phi = ' + str(np.round(phi,2)), transform = ax.transAxes, ha = 'left',
            va = 'top', fontsize = 'large')
    ax.text(0.65,0.78, 'z   = ' + str(np.round(z,2)), transform = ax.transAxes, ha = 'left',
            va = 'top', fontsize = 'large')

    # Customize RF Plot
    ax.set_xticks([0,R.shape[1]/2.,R.shape[1]])
    ax.set_yticks([0,R.shape[2]/2.,R.shape[2]])
    ax.set_xticklabels([-62.5,0,62.5], fontsize = 'large')
    ax.set_yticklabels([-47,0,47], fontsize = 'large')
    ax.set_xlabel('azimuth [deg]')
    ax.set_ylabel('elevation [deg]')
    sns.despine(ax = ax)
    ax.set_xlim([-5,R.shape[1]+5])
    ax.set_ylim([-5,R.shape[2]+5])
        
            
    with sns.axes_style("white"), sns.axes_style("ticks"):
        ax = plt.subplot(gs[:1,1])
        
        plt.plot(taus,
                 R[:,int(round(max_i)),int(round(max_j))],
                 'r')
        plt.plot(taus,
                 t_gauss,
                 'k')        
#         ax.set_xticks(np.linspace(taus[0],taus[-1],30,dtype='int'))
#         ax.set_xticklabels(np.arange(0,R_normalized.shape[2]+60,60)/60., fontsize = 'large')
        sns.despine(ax = ax)
        ax.set_xlabel('time [s]', fontsize = 'large')
        ax.set_ylabel('z-score', fontsize = 'large')
        plt.yticks(fontsize = 'large')
        ax.text(0.9,0.9, f'max = {np.round(taus[max_t]*1000,2)}ms', 
                transform = ax.transAxes, ha = 'left', va = 'top', fontsize = 'large')
        ax.axhline(0.0, color = 'k', linestyle = '--')
    plt.tight_layout()
    
    data = {'i' : centers_i,
            'j' : centers_j,
            'phi'      : phi,
            'zscore'   : z,
            'peak_tau' :round(taus[int(max_t)],2)
            }
    return fig,data

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