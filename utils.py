import pandas as pd
import numpy as np
from dlab.generalephys import get_waveform_duration,get_waveform_PTratio,get_waveform_repolarizationslope,option234_positions
from scipy.cluster.vq import kmeans2
import seaborn as sns;sns.set_style("ticks")
import matplotlib.pyplot as plt
import h5py
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import os

def get_peak_waveform_from_template(template):
    max = 0
    ind=0
    peak = np.zeros(np.shape(template.T)[0])
    for i,wv in enumerate(template.T):
        if np.max(np.abs(wv)) > max:
            max = np.max(np.abs(wv))
            ind = i
            peak = wv
    return peak

def df_from_phy(folder,expnum='1',recnum='1',site_positions = option234_positions,**kwargs):
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]

        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                print(base_folder)
                return None
        else:
            rec_folder = folder
        raw_path = os.path.join(rec_folder,'experiment'+str(expnum),'recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            print('loading from '+raw_path)
        else:
            
            print('could not find data folder for '+raw_path)
    
        if os.path.isfile(os.path.join(raw_path,'spike_clusters.npy')) :                  
    #             df = df_from_phy(raw_path,site_positions = ephys.option234_positions,cluster_file='KS2',cohort=cohort,mouse=mouse)

            path = raw_path
            units = ephys.load_phy_template(path,cluster_file='KS2',site_positions=site_positions)
            #structures is a dictionary that defines the bounds of the structure e.g.:{'v1':(0,850), 'hpc':(850,2000)}
            mouse = [];experiment=[];cell = [];ypos = [];xpos = [];waveform=[];template=[];structure=[];times=[]
            index = []; count = 1; cohort = []
            probe_id=[]
            depth=[];#print(list(nwb_data.keys()));print(list(nwb_data['processing'].keys()));

            for unit in list(units.keys()):
                if 'probe' in kwargs.keys():
                    probe_id.extend([kwargs['probe']])
                else:
                    probe_id.extend(['A'])
                if 'mouse' in kwargs.keys():
                    mouse.extend([kwargs['mouse']])
                else:
                    mouse.extend([mouse_])
                if 'experiment' in kwargs.keys():
                    experiment.extend([kwargs['experiment']])
                else:
                    experiment.extend(['placeholder'])
                if 'cohort' in kwargs.keys():
                    cohort.extend([kwargs['cohort']])
                else:
                    cohort.extend([cohort_])

                xpos.extend([units[unit]['xpos']])
                ypos.extend([units[unit]['ypos']])
                template.extend([units[unit]['template']])
                times.append(units[unit]['times'])
                waveform.append(units[unit]['waveform_weights'])

            df = pd.DataFrame(index=index)
            df = df.fillna(np.nan)
        #     df['nwb_id'] = nwb_id
            df['mouse'] = mouse
            df['experiment'] = experiment
            df['probe'] = probe_id
        #     df['structure'] = structure
            df['cell'] = units.keys()
            df['cohort'] = cohort
            df['times'] = times
            df['ypos'] = ypos
            df['xpos'] = xpos
    #         df['depth'] = depth
            df['waveform'] = waveform
            df['template'] = template
            return df

def df_from_nwb(nwb_data,structures=None,insertion_angle=55,nwbid=0):
    if type(nwb_data)==str:
        #print(nwb_data)
        nwbid = nwb_data
        nwb_data = h5py.File(nwb_data)
    else:
        nwb_data = nwb_data
    #structures is a dictionary that defines the bounds of the structure e.g.:{'v1':(0,850), 'hpc':(850,2000)}
    mouse = [];experiment=[];cell = [];ypos = [];xpos = [];waveform=[];template=[];structure=[];times=[]
    index = []; count = 1
    nwb_id = [];probe_id=[]
    depth=[];#print(list(nwb_data.keys()));print(list(nwb_data['processing'].keys()));
    if 'processing' in nwb_data.keys():
        for probe in list(nwb_data['processing'].keys()):
            if 'UnitTimes' in list(nwb_data['processing'][probe].keys()): 
                for i,u in enumerate(list(nwb_data['processing'][probe]['UnitTimes'].keys())):
                    if u != 'unit_list':
                        nwb_id.append(nwbid)
                        probe_id.append(probe)
                        index.append(count);count+=1
                        mouse.append(str(np.array(nwb_data.get('identifier'))))
                        experiment.append(1)
                        cell.append(u)
                        times.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['times']));# print(list(nwb_data['processing'][probe]['UnitTimes'][u].keys()))
                        if 'ypos' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                            ypos.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['ypos']))
                            has_ypos = True
                        else:
                            ypos.append(None)
                            has_ypos = False				
                        if 'depth' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                            depth.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['depth']))
                        else:
                            if has_ypos:
                                depth.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['ypos']))
                            else:
                                depth.append(None)
                        if 'xpos' in list(nwb_data['processing'][probe]['UnitTimes'][u].keys()):
                            xpos.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['xpos']))
                            has_xpos = True
                        else:
                            xpos.append(None)
                            has_xpos = False
                        template.append(np.array(nwb_data['processing'][probe]['UnitTimes'][u]['template']))
                        waveform.append(get_peak_waveform_from_template(template[-1]))
                        if not structures == None:
                            structur = None
                            for struct, bounds in structures.iteritems():
                                if ypos[-1] > bounds[0] and ypos[-1]< bounds[1] :
                                    structur=struct
                        else:
                            structur = None
                        structure.append(structur)
    df = pd.DataFrame(index=index)
    df = df.fillna(np.nan)
    df['nwb_id'] = nwb_id
    df['mouse'] = mouse
    df['experiment'] = experiment
    df['probe'] = probe_id
    df['structure'] = structure
    df['cell'] = cell
    df['times'] = times
    df['ypos'] = ypos
    df['xpos'] = xpos
    df['depth'] = depth
    df['waveform'] = waveform
    df['template'] = template
    return df

def classify_waveform_shape(df,plots=False,save_plots=False,basepath='',kmeans=0):
	durations = np.zeros(np.shape(df)[0])
	PTratio = np.zeros(np.shape(df)[0])
	repolarizationslope = np.zeros(np.shape(df)[0])
	for i,waveform in enumerate(df.waveform):
        # try:
		durations[i]=get_waveform_duration(waveform)
		PTratio[i]=get_waveform_PTratio(waveform)
		repolarizationslope[i]=get_waveform_repolarizationslope(waveform,window=18)
        # except:
        #     durations[i]=np.nan
        #     PTratio[i]=np.nan
        #     repolarizationslope[i]=np.nan
	df['waveform_duration'] = durations
	df['waveform_PTratio'] = PTratio
	df['waveform_repolarizationslope'] = repolarizationslope

	waveform_k = kmeans2(np.vstack(((durations-np.min(durations))/np.max((durations-np.min(durations))),
									(PTratio-np.min(PTratio))/np.max((PTratio-np.min(PTratio))),
									(repolarizationslope-np.min(repolarizationslope))/np.max((repolarizationslope-np.min(repolarizationslope))))).T,
							2, iter=300, thresh=5e-6,minit='points')
    # waveform_k = kmeans2(np.vstack((durations/np.max(durations),PTratio/np.max(PTratio))).T, 2, iter=300, thresh=5e-6,minit='points')
    # waveform_k = kmeans2(np.vstack((durations/np.max(durations),(repolarizationslope-np.min(repolarizationslope))/np.max(repolarizationslope))).T, 2, iter=900, thresh=5e-7,minit='points')
    
    #assign fs and rs to the kmeans results
	if np.mean(durations[np.where(waveform_k[1]==0)[0]]) < np.mean(durations[np.where(waveform_k[1]==1)[0]]):
		fs_k = 0;rs_k = 1
		waveform_class_ids = ['fs','rs']
	else:
		rs_k = 0;fs_k = 1
		waveform_class_ids = ['rs','fs']
	waveform_class = [waveform_class_ids[k] for k in waveform_k[1]]

	#uncomment this to ignore the preceding kmeans and just split on the marginal distribution of durations
	if kmeans==0:
		waveform_class = ['fs' if duration < 0.0004 else 'rs' for i,duration in enumerate(durations) ]
	else:
		waveform_k = kmeans2(np.vstack(((durations-np.min(durations))/np.max((durations-np.min(durations))),
										(PTratio-np.min(PTratio))/np.max((PTratio-np.min(PTratio))),
										(repolarizationslope-np.min(repolarizationslope))/np.max((repolarizationslope-np.min(repolarizationslope))))).T,
								kmeans, iter=300, thresh=5e-6,minit='points')
		# waveform_k = kmeans2(np.vstack((durations/np.max(durations),PTratio/np.max(PTratio))).T, 2, iter=300, thresh=5e-6,minit='points')
		# waveform_k = kmeans2(np.vstack((durations/np.max(durations),(repolarizationslope-np.min(repolarizationslope))/np.max(repolarizationslope))).T, 2, iter=900, thresh=5e-7,minit='points')
		
		#assign fs and rs to the kmeans results
		if np.mean(durations[np.where(waveform_k[1]==0)[0]]) < np.mean(durations[np.where(waveform_k[1]==1)[0]]):
			fs_k = 0;rs_k = 1
			waveform_class_ids = ['fs','rs']
		else:
			rs_k = 0;fs_k = 1
			waveform_class_ids = ['rs','fs']
		waveform_class = [waveform_class_ids[k] for k in waveform_k[1]]

	#force upwards spikes to have the own class, because we're not sure how they fit in this framework
	waveform_class = [waveform_class[i] if ratio < 1.0 else 'up' for i,ratio in enumerate(PTratio) ]
	df['waveform_class']=waveform_class

	#mark narrow upwards spikes as axons
	waveform_class = ['axon' if all([duration < 0.0004,waveform_class[i]=='up']) else waveform_class[i] for i,duration in enumerate(durations) ]
	df['waveform_class']=waveform_class

	# #mark narrow downward spike at the very bottom of cortex as axons
	#waveform_class = ['axon' if all([duration < 0.0004,waveform_class[i]=='fs',df['depth'][i+1] > 750, df['depth'][i+1]<1050]) else waveform_class[i] for i,duration in enumerate(durations) ]
	df['waveform_class']=waveform_class

	if plots:
		plot_waveform_classification(durations, PTratio, repolarizationslope,df,save_plots=save_plots,basepath=basepath)
	return df

def plot_waveform_classification(durations, PTratio, repolarizationslope, df,save_plots=False, basepath=''):
	f,ax = plt.subplots(1,3,figsize=(8,3))
	ax[0].plot(durations[np.where(df.waveform_class=='rs')[0]],PTratio[np.where(df.waveform_class=='rs')[0]],'o',ms=3.2)
	ax[0].plot(durations[np.where(df.waveform_class=='fs')[0]],PTratio[np.where(df.waveform_class=='fs')[0]],'o',ms=3.2)
	#ax[0].plot(durations[np.where(df.waveform_class=='up')[0]],PTratio[np.where(df.waveform_class=='up')[0]],'o',ms=3.2)
	ax[0].plot(durations[np.where(df.waveform_class=='axon')[0]],PTratio[np.where(df.waveform_class=='axon')[0]],'o',ms=3.2)
	ax[0].set_xlabel('width (sec)')
	ax[0].set_ylabel('peak/trough ratio')
	ax[1].plot(durations[np.where(df.waveform_class=='rs')[0]],repolarizationslope[np.where(df.waveform_class=='rs')[0]],'o',ms=3.2)
	ax[1].plot(durations[np.where(df.waveform_class=='fs')[0]],repolarizationslope[np.where(df.waveform_class=='fs')[0]],'o',ms=3.2)
	#ax[1].plot(durations[np.where(df.waveform_class=='up')[0]],repolarizationslope[np.where(df.waveform_class=='up')[0]],'o',ms=3.2)
	ax[1].plot(durations[np.where(df.waveform_class=='axon')[0]],repolarizationslope[np.where(df.waveform_class=='axon')[0]],'o',ms=3.2)
	ax[1].set_xlabel('width (sec)')
	ax[1].set_ylabel('repolarization slope')
	ax[2].plot(PTratio[np.where(df.waveform_class=='rs')[0]],repolarizationslope[np.where(df.waveform_class=='rs')[0]],'o',ms=3.2)
	ax[2].plot(PTratio[np.where(df.waveform_class=='fs')[0]],repolarizationslope[np.where(df.waveform_class=='fs')[0]],'o',ms=3.2)
	#ax[2].plot(PTratio[np.where(df.waveform_class=='up')[0]],repolarizationslope[np.where(df.waveform_class=='up')[0]],'o',ms=3.2)
	ax[2].plot(PTratio[np.where(df.waveform_class=='axon')[0]],repolarizationslope[np.where(df.waveform_class=='axon')[0]],'o',ms=3.2)
	ax[2].set_ylabel('repolarization slope')
	ax[2].set_xlabel('peak/trough ratio')
	ax[0].set_xlim(0.0,0.0015);ax[1].set_xlim(0.0,0.0015)
	ax[0].set_ylim(0,1.1);ax[2].set_xlim(0,1.1)
	plt.tight_layout()
	for axis in ax:
	#    ephys.cleanAxes(axis,bottomLabels=True,leftLabels=True)
		axis.locator_params(axis='x',nbins=4)
	ax[2].legend(loc='upper right')
	panelname = 'waveforms_clusters'
	plt.tight_layout()
	if save_plots:
		plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.png'),fmt='png',dpi=300)
		plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.eps'),fmt='eps')
		
	nbins = 36
	plt.hist(durations[np.where(df.waveform_class=='rs')[0]],range=(0,0.0015),bins=nbins)
	plt.hist(durations[np.where(df.waveform_class=='fs')[0]],range=(0,0.0015),bins=nbins)
	plt.hist(durations[np.where(df.waveform_class=='axon')[0]],range=(0,0.0015),bins=nbins)
	plt.figure()
	plt.hist((durations[np.where(df.waveform_class=='rs')[0]],durations[np.where(df.waveform_class=='fs')[0]],durations[np.where(df.waveform_class=='axon')[0]]),range=(0,0.0015),bins=nbins,stacked=True)
	#ephys.cleanAxes(plt.gca(),bottomLabels=True,leftLabels=True)
	plt.xlabel('waveform duration (sec)')
	plt.ylabel('neuron count')
	panelname = 'waveforms_durationhistogram'
	plt.tight_layout()
	if save_plots:
		plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.png'),fmt='png',dpi=300)
		plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.eps'),fmt='eps')
	
	plt.figure(figsize=(4,3))
	
	waveform_time = np.linspace(-1*np.where(df.waveform[1] > 0.)[0][0]/30000.,(len(df.waveform[1])-np.where(df.waveform[1] > 0.)[0][0])/30000.,len(df.waveform[1]))*1000
	#plot all
	for i,waveform in enumerate(df.waveform):
		#waveform_time = np.linspace(0,len(waveform)/30000.,len(waveform))*1000
		if df.waveform_class[i]=='rs':
			plt.plot(waveform_time,waveform/np.max(np.abs(waveform)),color=sns.color_palette()[0],alpha=0.01)
		if df.waveform_class[i]=='axon':#df.waveform_class.unique()[np.where(df.waveform_class=='axon')[0]]:
			plt.plot(waveform_time,waveform/np.max(np.abs(waveform)),color=sns.color_palette()[2],alpha=0.01)
		if df.waveform_class[i]=='fs':#df.waveform_class.unique()[np.where(df.waveform_class=='fs')[0]]:
			plt.plot(waveform_time,waveform/np.max(np.abs(waveform)),color=sns.color_palette()[1],alpha=0.01)
	# plot means, normalized
	for waveform_class in ['rs','fs','axon']:#df.waveform_class.unique():
		if waveform_class != 'up' and waveform_class!='axon':
			plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class==waveform_class])/(np.max(np.abs(np.mean(df.waveform[df.waveform_class==waveform_class])))),lw=4)
		#plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class==waveform_class])/(np.max(np.abs(np.mean(df.waveform[df.waveform_class==waveform_class])))),lw=2)
	# plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class=='rs'])/(np.min(np.mean(df.waveform[df.waveform_class=='rs']))*-1),lw=2)
	# plt.plot(waveform_time,np.mean(df.waveform[df.waveform_class=='up'])/(np.max(np.mean(df.waveform[df.waveform_class=='up']))),lw=2)
	
	plt.title('RS: '+str(len(df.waveform_class[df.waveform_class=='rs']))+
				'   FS: '+str(len(df.waveform_class[df.waveform_class=='fs']))+
			  '   axon: '+str(len(df.waveform_class[df.waveform_class=='axon'])))#+
	#            '   up:'+str(len(df.waveform_class[df.waveform_class=='up'])))
	
	
	plt.gca().set_xlim(-1.,1.4)
	plt.gca().legend(loc='upper left')
	#ephys.cleanAxes(plt.gca(),leftLabels=True,bottomLabels=True)
	plt.gca().set_ylabel('normalized amplitude',size=10)
	d=plt.gca().set_xlabel('time (msec)',size=10)
	panelname = 'waveforms_mean_peak'
	plt.tight_layout()
	if save_plots:
		plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.png'),fmt='png',dpi=300)
		plt.gcf().savefig(os.path.join(basepath,'figures','panels',panelname+'.eps'),fmt='eps')

def drawPhaseIIIProbe(colors,ax=-1,highlight=-1,clim=None, cmap='viridis', drawLines=False):
	'''
	Args:
		colors: a list of values to plotted as colors on the probe
		ax
		highlight
		clim: color map limits
		cmap: color map to use; default viridis
		drawLines: whether or not to draw the outline of the probe; default is False
	Returns:
		None, plots an image of the input colors on a Phase3A Neuropixels probes
	written by josh siegle
	'''
	if ax == -1:
		fig, ax = plt.subplots()
	
	patches = []
	
	for ch in range(0,len(colors)):
		
		channelPos = ch % 4
		channelHeight = ch / 4
		
		if channelPos == 0:
			xloc = -1.5
			yloc = channelHeight*2
		elif channelPos == 1:
			xloc = 0.5
			yloc = channelHeight*2
		elif channelPos == 2:
			xloc = -0.5
			yloc = channelHeight*2 + 1
		else:
			xloc = 1.5
			yloc = channelHeight*2 + 1
	
		rect = mpatches.Rectangle([xloc, yloc], 1.0, 2.0, ec="none", ls='None')
		
		if drawLines:
			if ch % 50 == 0:
				plt.plot([-5, 6], [yloc, yloc], 'gray')
				
			if ch % 100 == 0:
				plt.plot([-5, 6], [yloc, yloc], '-k')
			
		patches.append(rect)
		
		if ch == highlight:
			highlightX = xloc
			highlightY = yloc
			highlight = 1
		
	collection = PatchCollection(patches, cmap=cmap)
	
	collection.set_array(colors)
	if clim != None:
		collection.set_clim(clim[0],clim[1])
	ax.add_collection(collection)
	
	for ch in np.arange(0,len(colors),50):
		plt.plot([-2.5,-2],[ch/2, ch/2],'k')
	
	
	if highlight > -1:
		print(highlightY)
		plt.plot(highlightX, highlightY, color=[1,1,1])
	
	plt.axis('off')
	plt.xlim((-5,6))
	plt.ylim((-5,ch/2 + 20))

def get_spike_limits(nwb_data):
    firsts = [np.array(nwb_data['processing'][nwb_data['processing'].keys()[0]]['UnitTimes'][other]['times'])[0]\
              for other in np.array(nwb_data['processing'][nwb_data['processing'].keys()[0]]['UnitTimes']['unit_list'])]
    lasts = [np.array(nwb_data['processing'][nwb_data['processing'].keys()[0]]['UnitTimes'][other]['times'])[-1]\
              for other in np.array(nwb_data['processing'][nwb_data['processing'].keys()[0]]['UnitTimes']['unit_list'])]
    return np.min(firsts),np.max(lasts)