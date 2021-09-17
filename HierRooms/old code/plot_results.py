import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# name = "HierarchicalRooms"
# name = "HierarchicalRooms_no_max_particles"
# name = "HierarchicalRooms_min_particle_70"
# name = "HierarchicalRooms_likelihood_cap_0"
# name = "HierarchicalRooms_indep"
# name = "HierarchicalRooms_joint"
# name = "HierarchicalRooms_indep3"
# name = "HierarchicalRooms_joint2"
name = "HierarchicalRooms_indep_no_structure"


results = pd.read_pickle("./analyses/"+name+".pkl")


sns.set_context('paper', font_scale=1.25)
X0 = results[results['Success'] == True]
from matplotlib import gridspec

# sns.set_context('talk')
# with sns.axes_style('ticks'):
#     fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
#     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
#                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
#                    )
#     ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
#     ax.plot([-0.5, 4], [ybar, ybar], 'r--')
#     ax.set_ylabel('Total Steps')
#     ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

#     sns.despine()
#     plt.savefig("figs/"+name+'_violin.png', dpi=300, bbox_inches='tight')


with sns.axes_style('ticks'):
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(8, 3)) 
    gs = gridspec.GridSpec(1, 3, width_ratios=[2.0, 1, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    # plot histogram of cumulative steps
    sns.distplot(X0[X0['Model']=='Hierarchical']['Cumulative Steps'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X0[X0['Model']=='Independent']['Cumulative Steps'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X0[X0['Model']=='Flat']['Cumulative Steps'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlabel('Cumulative Steps')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)
    # xlabels = ['{:,.1E}'.format(x) for x in ax0.get_xticks()]
    # ax0.set_xticklabels(xlabels)
    # ax0.ticklabel_format(axis='x', style='sci',scilimit=(0,0))




    # plot bar chart of cumulative steps
    h_sims_hc = len(X0[X0['Model']=='Hierarchical'])
    h_sims_ic = len(X0[X0['Model']=='Independent'])
    h_sims_fl = len(X0[X0['Model']=='Flat'])
    
    X1 = pd.DataFrame({
        'Cumulative Steps Taken': np.concatenate([
                X0[X0['Model']=='Hierarchical']['Cumulative Steps'].values,
                X0[X0['Model']=='Independent']['Cumulative Steps'].values,
                X0[X0['Model']=='Flat']['Cumulative Steps'].values, 
            ]),
        'Model': ['Hierarchical'] * h_sims_hc + ['Independent'] * h_sims_ic + ['Flat'] * h_sims_fl,
    })
    sns.barplot(data=X1, x='Model', y='Cumulative Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Total Steps')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    # ax1.ticklabel_format(axis='y', style='sci',scilimit=(0,0))
    # ylabels = ['{:,.1E}'.format(y) for y in ax1.get_yticks()]
    # ax1.set_yticklabels(ylabels)



    # plot violin plot of cumulative steps
    sns.violinplot(data=X0, x='Model', y='Cumulative Steps', ax=ax2, palette='Dark2',
                    order=["Flat", "Independent", "Hierarchical"]
                    )
    ax2.set_ylabel('Total Steps')
    ax2.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    # ax2.ticklabel_format(axis='y', style='sci',scilimit=(0,0))
    # ylabels = ['{:,.1E}'.format(y) for y in ax2.get_yticks()]
    # ax2.set_yticklabels(ylabels)
    sns.despine()

    plt.tight_layout()
    plt.savefig("figs/"+name+'.png', dpi=300, bbox_inches='tight')