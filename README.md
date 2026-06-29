# denmanlab
denman lab shared functions, scripts, and pipelines

Denman Lab
University of Colorado Anschutz Medical Campus

examples:
```
from denmanlab import psth_and_raster as psth

unit=239

f,ax=plt.subplots(1,1)
psth.psth_line(times=df1[df1.index==unit].spike_times.values[0],
         triggers=df_stim[(df_stim.stimulus=='luminance_flash') & (df_stim.optogenetics_LED_state == 0)].start_time.values,
              ymax=40,binsize=0.05,axes=ax,color='#487697')
psth.psth_line(times=df1[df1.index==unit].spike_times.values[0],
         triggers=df_stim[(df_stim.stimulus=='luminance_flash') & (df_stim.optogenetics_LED_state == 1)].start_time.values,
              ymax=40,binsize=0.05,axes=ax,color='#ffaa

f,ax=plt.subplots(2,1)
psth.raster(times=df1[df1.index==unit].spike_times.values[0],
         triggers=df_stim[(df_stim.stimulus=='luminance_flash') & (df_stim.optogenetics_LED_state == 0)].start_time.values,
              axes=ax[0],color='#487697',timeDomain=True,post=1.5,ms=8)
psth.raster(times=df1[df1.index==unit].spike_times.values[0],
         triggers=df_stim[(df_stim.stimulus=='luminance_flash') & (df_stim.optogenetics_LED_state == 1)].start_time.values,
              axes=ax[1],color='#ffaa00',timeDomain=True,post=1.5,ms=8)
for ax_ in ax: ax_.set_xlim(-0.5,1.0)
plt.tight_layout()

<img width="622" height="969" alt="Screenshot 2026-06-29 at 10 55 42" src="https://github.com/user-attachments/assets/e610ccc7-0bef-44c0-bd0d-d81a89facf19" />

```
```
from denmanlab import rf_analysis as rf


sta = rf.sta(spiketimes=df_units[df_units.index==unit].spike_times.values[0],
             data=m.T,
             datatimes=matrix_times,
             taus = taus,
             samplingRateInkHz=1)
rf.plotsta(sta,taus,colorrange=(np.min(sta['-0.01']),np.max(sta['-0.01'])),smooth=.6)

<img width="982" height="459" alt="Screenshot 2026-06-29 at 10 55 28" src="https://github.com/user-attachments/assets/bec25402-fa63-4789-be51-deee5840cae8" />

```

