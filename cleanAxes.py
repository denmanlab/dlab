def cleanAxes(ax,bottomLabels=False,leftLabels=False,rightLabels=False,topLabels=False,total=False):
    ax.tick_params(axis='both',labelsize=10)
    ax.spines['top'].set_visible(False);
    ax.yaxis.set_ticks_position('left');
    ax.spines['right'].set_visible(False);
    ax.xaxis.set_ticks_position('bottom')
    if not bottomLabels or topLabels:
        ax.set_xticklabels([])
    if not leftLabels or rightLabels:
        ax.set_yticklabels([])
    if rightLabels:
        ax.spines['right'].set_visible(True);
        ax.spines['left'].set_visible(False);
        ax.yaxis.set_ticks_position('right');
    if total:
        ax.set_frame_on(False);
        ax.set_xticklabels('',visible=False);
        ax.set_xticks([]);
        ax.set_yticklabels('',visible=False);
        ax.set_yticks([])