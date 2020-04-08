import numpy as np
import os,sys,glob
import pandas as pd
from scipy.stats import pearsonr, spearmanr, zscore

def pairwise_correlation(df,col,method='pearson',z_score=True):
    output_matrix = np.zeros((df.shape[0],df.shape[0]))
    output_matrix[:,:] = np.nan
    for i,d1 in df[col].iteritems():
        for j,d2 in df[col].iteritems():
            if z_score:
                d1 = zscore(d1)
                d2 = zscore(d2)
            try:
                if method =='pearson' :
                    output_matrix[i-df.index[0]][j-df.index[0]]= pearsonr(d1,d2)[0]
                if method == 'spearman':
                    output_matrix[i][j]= spearmanr(d1,d2)[0]
            except: output_matrix[i-df.index[0]][j-df.index[0]] = np.nan
    return output_matrix

def pairwise_df(df):
    '''
    input: a df of single cells, with cell as a field
    returns a new df with all the pairwise combinations
    '''
    df_pairs = pd.DataFrame(index=range(total_pairs),
                        columns=['mouse','ind1','ind2','cell1','cell2','type1','type2')
    start_ = 0
    for mouseid in df.mouse.unique():
        df_=df[df.mouse==mouseid]
        numpairs=len(list(combinations(df_.cell,2)))
        
        df_pairs['mouse'][start_:start_+numpairs]=mouseid
        df_pairs['cell1'][start_:start_+numpairs]=[c[0] for c in list(combinations(df_.cell,2))]
        df_pairs['cell2'][start_:start_+numpairs]=[c[1] for c in list(combinations(df_.cell,2))]
        df_pairs['ind1'][start_:start_+numpairs]=[c[0] for c in list(combinations(df_.index,2))]
        df_pairs['ind2'][start_:start_+numpairs]=[c[1] for c in list(combinations(df_.index,2))]
        df_pairs['type1'][start_:start_+numpairs]=[c[0] for c in list(combinations(df_.waveform_class,2))]
        df_pairs['type2'][start_:start_+numpairs]=[c[1] for c in list(combinations(df_.waveform_class,2))]

        start_+=numpairs

def make_tensor(df,time,duration,binsize):
    #mak
    cell_ = rp.get_binned(df.times[df.index[0]],[time],pre=0.,post=duration,binsize=binsize)
    tensor = np.zeros((len(cell_[0]),df.shape[0]))
    ind=0
    for t in df.times:
        cell_ = rp.get_binned(t,[time],pre=0.,post=duration,binsize=binsize)
        tensor[:,ind] = cell_[0]
        ind+=1
    return tensor

def dimensionality_PCA(ten):
    ev=[]
    for i in range(min(ten.shape[0], ten.shape[1])-1):
        PCA = decomposition.PCA(n_components=i)
        f=PCA.fit(ten)
        ev.append(sum(PCA.explained_variance_))
    return np.where(ev>np.max(ev)*.95)[0][0]