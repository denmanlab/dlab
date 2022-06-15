from brainrender import Scene
from brainrender import settings
from brainrender.actors import Points
settings.SHOW_AXES = False
settings.WHOLE_SCREEN = False

import numpy as np
from matplotlib import cm
from matplotlib.colors import to_hex
import pandas as pd

#load data
path = '/Users/danieljdenman/Desktop/crossings.csv'
df = pd.read_csv(path).transpose()
df.columns = df.iloc[0]

#calculate channel coords relative to an insertion origin
#assumes 45º angle insertion (to the horizontal plane), and 120º angle between each (in the horizontal plane). 
channels_in = 300
x=700
x_ = channels_in*10 / np.sqrt(2)
a=(3000-x*np.sqrt(2))/np.sqrt(2)
stim_electrode = [np.zeros(channels_in),
                  np.zeros(channels_in),
                  np.linspace(500,0,channels_in),
                  ]
probeA         = [np.linspace(a/2. * np.sqrt(3),-(x/2.)*np.sqrt(3),channels_in),
                  np.linspace(x_,0,channels_in),
                  np.linspace(-a/2.,(x/2.),channels_in),
                  ]
probeB         = [np.linspace(-a/2. * np.sqrt(3),(x/2.)*np.sqrt(3),channels_in),
                  np.linspace(x_,0,channels_in),
                  np.linspace(-a/2.,(x/2.),channels_in),
                  ]
probeC         = [np.ones(channels_in),
                  np.linspace(x_,0,channels_in),
                  np.linspace(x_-x,-x,channels_in),
                  ]

#create brainrender scene and add relevant areas using Allen CCF names
scene = Scene()
scene.add_brain_region("VISp",alpha=0.1)
scene.add_brain_region("VISpl",alpha=0.1)
scene.add_brain_region("HIP",alpha=0.1)
scene.add_brain_region("SUB",alpha=0.1)
scene.add_brain_region("POST",alpha=0.1)
scene.add_brain_region("ENT",alpha=0.1)
scene.add_brain_region("SCs",alpha=0.1)
scene.add_brain_region("SCm",alpha=0.1)

#add the data from each channel as a colored point
insertion_origin = np.array([9000,400,3200]) #[AP,DV,ML]
data = ['jlh6a','jlh6b','jlh6c']
mP = cm.get_cmap('inferno',50)
for i,p in enumerate([probeA,probeB,probeC]):
    cs = [to_hex(mP(c/5000.)) for c in df[data[i]].values[1:].astype(float)]
    coords = np.array([c+insertion_origin for c in np.array(p).T.reshape(-1,3)])
    scene.add(Points(coords,colors=cs))

#render with specified view. to change the view, can interact with plot then press 'c'. copy the params printed to the console and paste them over the ones in custom_camera
scene.content
custom_camera =  {
     'pos': (-14214, 1547, 13938),
     'viewup': (0, -1, 0),
     'clippingRange': (10986, 50720),
     'focalPoint': (8645, 3895, -3625),
     'distance': 28923,
   }
scene.render(camera=custom_camera,zoom=2.)
