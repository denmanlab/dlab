#utility for parsing the jstim color exchange pkl so only serializable objects are in it, and saaving the DataFrame separately
#call from directory where file is stored   

import pandas as pd
import os,glob
import _pickle as pkl

folder = os.getcwd()
a = pd.read_pickle(open(glob.glob(os.path.join(folder,'*_color_exchange.pkl'))[0],'rb'))
b={}
b['green'] = a['green'] 
b['uv'] = a['uv']
b['dropped_frames'] = a['dropped_frames'] 

a['contrasts'].to_csv(glob.glob(os.path.join(folder,'*_color_exchange.pkl'))[0].replace('pkl','csv'))
pkl.dump(b,open(glob.glob(os.path.join(folder,'*_color_exchange.pkl'))[0].replace('.pkl','_s.pkl'),'wb'))
