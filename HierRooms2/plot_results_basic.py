import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set_context('paper', font_scale=1.25)
from matplotlib import gridspec


# name = "indep"
# name = "joint"
# name = "ambig"
name = "indep_no_structure"


results = pd.read_pickle("./analyses/HierarchicalRooms_"+name+".pkl")

X0 = results[results['Success'] == True]


with sns.axes_style('ticks'):
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X0[X0['Model']=='Hierarchical']['Cumulative Steps'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X0[X0['Model']=='Independent']['Cumulative Steps'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X0[X0['Model']=='Flat']['Cumulative Steps'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlabel('Total Steps')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)




    # plot bar chart of cumulative steps
    sns.barplot(data=X0, x='Model', y='Cumulative Steps', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Total Steps')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    
    plt.tight_layout()
    plt.savefig("figs/"+name+'.png', dpi=300, bbox_inches='tight')

