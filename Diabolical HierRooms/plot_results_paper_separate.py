#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 03:30:52 2020

@author: rex
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_one_result(name, file_idx, savefig):
    results = pd.read_pickle("./analyses/HierarchicalRooms_"+name+".pkl")
    X0 = results[results['Success'] == True]

    sns.set_context('talk')

    # plot histogram of cumulative steps
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        cc = sns.color_palette('Dark2')

        sns.distplot(X0[X0['Model']=='Hierarchical']['Cumulative Steps'], label='Hier.', color=cc[2])
        sns.distplot(X0[X0['Model']=='Independent']['Cumulative Steps'], label='Ind.', color=cc[1])
        sns.distplot(X0[X0['Model']=='Flat']['Cumulative Steps'], label='Flat', color=cc[0])

        if file_idx==0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)

        ax.set_yticks([])
        ax.set_xlim([0, ax.get_xlim()[1] ])
        ax.set_xticks(range(0,int(ax.get_xlim()[1]),100000))
        ax.set_xlabel('Total Steps')
        
        sns.despine(offset=2)    
        ax.spines['left'].set_visible(False)

        plt.tight_layout()
        if savefig:
            plt.savefig("figs/HierRooms_"+name+'_histo.png', dpi=300, bbox_inches='tight')    


    # plot bar chart of cumulative steps
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        sns.barplot(data=X0, x='Model', y='Cumulative Steps', 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

        plt.tight_layout()
        if savefig:
            plt.savefig("figs/HierRooms_"+name+'_bar.png', dpi=300, bbox_inches='tight')    





name_list = ["indep", "joint", "ambig"]
n_files = len(name_list)

savefig = True
fig_size = (5, 4.5)

for idx, name in enumerate(name_list):
    print name
    
    plot_one_result(name, idx, savefig)
