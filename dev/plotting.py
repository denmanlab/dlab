import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.ndimage as ndimage

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
        
