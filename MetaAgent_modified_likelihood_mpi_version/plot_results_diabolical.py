#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 00:24:33 2020

@author: rex
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

name = "DiabolicalRoomsResults"


results = pd.read_pickle("./analyses/"+name+".pkl")

n_sims = results['Iteration'].max()+1

sns.set_context('paper', font_scale=1.25)
X0 = results[results['In Goal']].groupby(['Model', 'Iteration']).sum()
from matplotlib import gridspec

with sns.axes_style('ticks'):
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(6, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    sns.distplot(X0.loc['Flat']['Step'], label='Flat', ax=ax0, color=cc[0])
    sns.distplot(X0.loc['Independent']['Step'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X0.loc['Joint']['Step'], label='Joint', ax=ax0, color=cc[2])
    sns.distplot(X0.loc['Hierarchical']['Step'], label='Hier.', ax=ax0, color=cc[3])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlabel('Cumulative Steps')
    
    X1 = pd.DataFrame({
        'Cumulative Steps Taken': np.concatenate([
                X0.loc['Joint']['Step'].values,
                X0.loc['Independent']['Step'].values,
                X0.loc['Hierarchical']['Step'].values, 
                X0.loc['Flat']['Step'].values, 
            ]),
        'Model': ['Joint'] * n_sims + ['Independent'] * n_sims + ['Hierarchical'] * n_sims + ['Flat'] * n_sims,
    })
    sns.barplot(data=X1, x='Model', y='Cumulative Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Joint', 'Hierarchical'])
    ax1.set_ylabel('Total Steps')
    ax1.set_xticklabels(['Flat', 'Ind.', 'Joint', 'Hier.'])

    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)

    plt.tight_layout()
    fig.savefig('./figs/RoomsResults.png', dpi=300)
    
    
    
# df0 = results[results['In goal']].groupby(['Model', 'Simulation Number', 'Trial Number']).mean()
# df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
# df0 = df0.rename(index=str, columns={'n actions taken': "Cumulative Steps Taken"})
# df0 = df0[df0['Trial Number'] == df0['Trial Number'].max()]
# # sim1.groupby(['Model', 'Simulation Number']).mean()
# print df0.groupby('Model').mean()['Cumulative Steps Taken']
# print df0.groupby('Model')['Cumulative Steps Taken'].std()