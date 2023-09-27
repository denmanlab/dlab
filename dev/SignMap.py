import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
import tifffile as tiff
from glob import glob
from tqdm.notebook import tqdm

from scipy.ndimage import gaussian_filter, gaussian_filter1d,rotate
from scipy.fft import fftn, fftfreq, rfftn,fft
from scipy.stats import zscore, mode

try:
    from NeuroAnalysisTools import RetinotopicMapping as rm
except ImportError:
    print('NeuroAnalysisTools not installed. https://github.com/zhuangjun1981/NeuroAnalysisTools/tree/master/NeuroAnalysisTools')

from PIL import Image,ImageDraw


def _parse_binary_fname(fname,
                        lastidx=None,
                        dtype = 'uint16',
                        shape = None,
                        sep = '_'):
    '''
    Gets the data type and the shape from the filename 
    This is a helper function to use in load_dat.
    
    out = _parse_binary_fname(fname)
    
    With out default to: 
        out = dict(dtype=dtype, shape = shape, fnum = None)
    '''
    fn = os.path.splitext(os.path.basename(fname))[0]
    fnsplit = fn.split(sep)
    fnum = None
    if lastidx is None:
        # find the datatype first (that is the first dtype string from last)
        lastidx = -1
        idx = np.where([not f.isnumeric() for f in fnsplit])[0]
        for i in idx[::-1]:
            try:
                dtype = np.dtype(fnsplit[i])
                lastidx = i
            except TypeError:
                pass
    if dtype is None:
        dtype = np.dtype(fnsplit[lastidx])
    # further split in those before and after lastidx
    before = [f for f in fnsplit[:lastidx] if f.isdigit()]
    after = [f for f in fnsplit[lastidx:] if f.isdigit()]
    if shape is None:
        # then the shape are the last 3
        shape = [int(t) for t in before[-3:]]
    if len(after)>0:
        fnum = [int(t) for t in after]
    return dtype,shape,fnum

def mmap_dat(self,
                filename,
                mode = 'r',
                nframes = None,
                shape = None,
                dtype='uint16'
                ):
    '''
    Loads frames from a binary file as a memory map.
    This is useful when the data does not fit to memory.
    
    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        mode (str)           : memory map access mode (default 'r')
                'r'   | Open existing file for reading only.
                'r+'  | Open existing file for reading and writing.                 
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16) 
    Returns:
        A memory mapped  array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = mmap_dat(filename)
    '''
    
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None: # try to get it from the filename
        dtype,shape,_ = self._parse_binary_fname(filename,
                                            shape = shape,
                                            dtype = dtype)
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype
    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
    dt = np.dtype(dtype)
    return np.memmap(filename,
                    mode=mode,
                    dtype=dt,
                    shape = (int(nframes),*shape))
    
def load_dat(filename,
             nframes = None,
             offset  = 0,
             shape   = None,
             dtype   = 'uint16'
             ):
    '''
    Loads frames from a binary file.
    
    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16) 
    Returns:
        An array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = load_dat(filename)
    '''    
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None: # try to get it from the filename
        dtype,shape,_ = _parse_binary_fname(filename,
                                            shape = shape,
                                            dtype = dtype)
    
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype

    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
    framesize = int(np.prod(shape))

    offset = int(offset)
    with open(filename,'rb') as fd:
        fd.seek(offset*framesize*int(dt.itemsize))
        buf = np.fromfile(fd,dtype = dt, count=framesize*nframes)
    buf = buf.reshape((-1,*shape),
                    order='C')
    buf = np.squeeze(buf)
    # buf = rotate(buf,axes=(1,2),angle=-90)
    return buf

def beer_lambert(stack,blFrames):
    bl        = np.mean(stack[:blFrames,:,:],axis=0)
    corrected = np.log(stack[blFrames:,:,:]/bl[np.newaxis,:,:])
    
    return corrected

def normalize(stack,blFrames):
    '''
    Z-transforms the stack.
    '''
    bl        = np.mean(stack[:blFrames,:,:],axis=0)
    corrected = stack[blFrames:,:,:]-bl[np.newaxis,:,:]
    
    return corrected

def zhuang(stack):
    smooth_data    = gaussian_filter(stack, sigma=(5), axes=(0))
    spectrum_movie = fftn(smooth_data, axes=(0))
    
    #Generate power movie
    power_movie = (np.abs(spectrum_movie)*2.)/len(stack)
    power_map   = np.abs(power_movie[1,:,:])
    
    #Generate phase movie
    phase_movie = np.angle(spectrum_movie)
    phase_map   = -1 * phase_movie[1,:,:]
    phase_map   = phase_map % (2*np.pi)
    
    return power_map, phase_map

class SignMap:
    def __init__(self,
                 recording_path,
                 fr = 10,
                 baseline=2,
                 dark=5
                 ):
        self.recording_path = recording_path
        self.baseline       = baseline
        self.dark           = dark
        
        self.files      = glob(os.path.join(recording_path,'Frames*.dat'))
        self.firstTrial = load_dat(self.files[0])
        self.nImg       = len(self.files)
        
        self.blFrames   = baseline*fr
        self.darkFrames = dark*fr
        self.stimFrames = self.firstTrial.shape[0] - self.blFrames - self.darkFrames
        
        stim_info_path = glob(os.path.join(recording_path,'*.pkl'))[0]
        stim_pkl       = pkl.load(open(stim_info_path,'rb'))
        stim_table     = [stim_pkl['bgsweeptable'][i] for i in stim_pkl['bgsweeporder']]
        self.stim_info = pd.DataFrame(stim_table,columns=['contrast','PosY','TF','SF','phase','PosX','grating_orientation','aperture_orientation'])
        
        
        print(f'{self.firstTrial.shape[0]} FRAMES, HEIGHT: {self.firstTrial.shape[1]}, WIDTH: {self.firstTrial.shape[2]}')
        
        print(f'{self.nImg} images (trials) found')
    
    def open_snapshots(self):
        '''
        Opens the snapshots from the recording path.
        '''
        snapshots = glob(os.path.join(self.recording_path,'Snapshot','*.jpg'))
        snapshots.sort()
        
        img1              = Image.open(snapshots[0])
        snapshot_array    = np.zeros((len(snapshots),*img1.shape))
        snapshot_array[0] = np.array(img1)
        
        if len(snapshots) > 1:
            for i, img in enumerate(snapshots[1:]):
                snapshot_array[i+1] = np.array(Image.open(img))
            
        return snapshot_array
    
    def pixel_distribution(self):
        '''
        Displays the pixel distribution of the recording.
        '''
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        
        ax[0].imshow(rotate(self.firstTrial[0,:,:],-90))
        ax[1].hist(self.firstTrial[80,:,:].flatten(),bins=50)
        
        plt.tight_layout()
        plt.show()
        

    
    def avg_phase(self,correction='beer lambert'):
        
        frameH = self.firstTrial.shape[0]
        frameW = self.firstTrial.shape[1]
        
        #precompute values from stim info
        grating_orientations  = self.stim_info['grating_orientation'].values
        aperture_orientations = self.stim_info['aperture_orientation'].values
        
        #initialize maps
        maps   = [np.zeros((frameH, frameW)) for _ in range(8)]
        counts = np.zeros((8))
        
        for i, path in tqdm(enumerate(self.files),total=self.nImg):
            
            orientation_idx = 0 if grating_orientations[i]  < 1.0 else 1
            aperture_idx    = 0 if aperture_orientations[i] < 1.0 else 2
            desired_count   = orientation_idx + aperture_idx
            
            stack = load_dat(path).astype('float')
            
            if correction == 'beer lambert':
                stack = beer_lambert(stack,self.blFrames)
            else:
                stack = normalize(stack,self.blFrames)
                
            power_map, phase_map = zhuang(stack)
            
            phasemapvec                                          = phase_map.copy()
            phasemapvec[(phasemapvec<0.25) & (phasemapvec>0.25)] = np.nan
            var1                                                 = np.nanvar(phasemapvec)
            phasemapvec[np.where(phasemapvec<0)]                 = phasemapvec[np.where(phasemapvec<0)]+2*np.pi
            varphase                                             = min(np.nanvar(phasemapvec),var1)

            if varphase >= 0.6:
                maps[desired_count]     += phase_map
                maps[desired_count+4]   += power_map
                counts[desired_count]   += 1
                counts[desired_count+4] += 1
            
        maps = [maps[i] / counts[i] for i in range(8)]            
            
        azimuth_phase   = (maps[0] - maps[1]) / 2
        elevation_phase = (maps[2] - maps[3]) / 2

        azimuth_power   = (maps[4] - maps[5]) / 2
        elevation_power = (maps[6] - maps[7]) / 2
        
        return [azimuth_phase, elevation_phase, azimuth_power, elevation_power]
    
    
            