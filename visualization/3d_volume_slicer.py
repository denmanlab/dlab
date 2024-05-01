"""Slice multiple datasets"""
from vedo import Plotter, Text2D, load, dataurl, ScalarBar3D, Volume

volumes = [dataurl+'vase.vti', dataurl+'embryo.slc', dataurl+'head.vti']
volumes = load(volumes)
cmaps = ['hot_r', 'gist_ncar_r', 'bone_r']

import numpy as np
rand_gen = np.random.default_rng(222)
#x = y = z = np.linspace(-2, 2, 41)
#X, Y, Z = np.meshgrid(x, y, z)
#values = 2*X*X - Y*Y + 1/(Z*Z+1)
#values = #

w1 = np.load('/home/nickg/asdf/spatiotemporal_receptive_data.npy')

#a = load(dataurl+'vase.vti') #Volume(values)
#b = load(dataurl+'vase.vti').permute_axes(0, 1, 2) #Volume(values).permute_axes(1, 2, 0)
#c = load(dataurl+'vase.vti').permute_axes(0, 2, 1) #Volume(values).permute_axes(2, 1, 0)
#volumes = [a, b, c]
time = 0
x = 1
y = 2
channels = 3
filters = 4

#volume1: x, y, time
#volume2: 
#volume3: 
volumes = [Volume(w1[:,:,:,:,0].mean(axis=3)).permute_axes(x, y, time), Volume(w1[:,:,:,:,0].mean(axis=3)).permute_axes(time, x, y), Volume(w1[:,:,:,:,0].mean(axis=3)).permute_axes(x, time, y)]
cmaps = ['PiYG', 'PiYG', 'PiYG']

#print(cmaps)

#exit()
########################################################################
def initfunc(iren, vol):

    vol.mode(1).cmap('k').alpha([0, 0, 0.15, 0, 0])
    txt = Text2D(data.filename[-20:], font='Calco')
    plt.at(iren).show(vol, vol.box(), txt)

    def func(widget, event):
        zs = int(widget.value)
        widget.title = f"z-slice = {zs}"
        msh = vol.zslice(zs)
        msh.cmap(cmaps[iren]).lighting("off")
        msh.name = "slice"
        sb = ScalarBar3D(msh, c='k')
        # sb = sb.clone2d("bottom-right", 0.08)
        plt.renderer = widget.renderer  # make it the current renderer
        plt.remove("slice", "ScalarBar3D").add(msh, sb)

    return func  # this is the actual function returned!


########################################################################
plt = Plotter(shape=(1, len(volumes)), sharecam=False, bg2='lightcyan')

for iren, data in enumerate(volumes):
    plt.add_slider(
        initfunc(iren, data), #func
        0, data.dimensions()[2],
        value=0,
        show_value=False,
        pos=[(0.1,0.1), (0.25,0.1)],
    )

plt.interactive().close()