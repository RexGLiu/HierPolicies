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


def plot_results(results, list_jointness):
    df0 = results[results['In Goal']].groupby(['Model', 'Iteration', 'Trial Number']).mean()
    df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
    df0 = df0.rename(index=str, columns={'Steps Taken': "Cumulative Steps Taken"})
    df0 = df0[df0['Trial Number'] == df0['Trial Number'].max()]
    print df0.groupby(['Model']).mean()['Cumulative Steps Taken']
    print df0.groupby(['Model'])['Cumulative Steps Taken'].std()

    in_goal = results[results['In Goal'] ].copy()
    in_goal['Contexts'] = [None] * len(in_goal)
    in_goal.loc[in_goal.Context < 5, 'Contexts'] = 'Training'
    in_goal.loc[in_goal.Context >= 5, 'Contexts'] = 'Test'


    print "plotting 1"
    with sns.axes_style('white'):
        g = sns.factorplot(y='Reward', data=in_goal, x='Contexts', 
                    hue='Model', units='Iteration', kind='bar', 
                    estimator=np.mean, palette='Accent', size=4)
        sns.despine(offset=5, trim=False)
        ax = g.axes[0][0]
        ax.set_ylabel('Average reward per trial')
        plt.savefig("figs/"+name+'_rewards.png', dpi=300, bbox_inches='tight')
    
    print "plotting 2"
    with sns.axes_style('ticks'):
        g = sns.factorplot(x='Context', y='Reward', data=in_goal[in_goal['Contexts'] == 'Test'], 
                    kind='bar', palette='Set2', col='Model', col_wrap=5, units='Iteration')
        g.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
        sns.despine(offset=10)
        plt.tight_layout()
        plt.savefig("figs/"+name+'_T.png', dpi=300, bbox_inches='tight')
    

name_list = ["Gen2GenEnvResults_fl", "Gen2GenEnvResults_ic", "Gen2GenEnvResults_jc", "Gen2GenEnvResults_m", 
             "Gen2GenEnvResults_hc_0", "Gen2GenEnvResults_hc_1", "Gen2GenEnvResults_hc_2", "Gen2GenEnvResults_hc_3", "Gen2GenEnvResults_hc_4", "Gen2GenEnvResults_hc_5"]

list_jointness = np.arange(0.,1.2,0.5)
list_jointness[0] = 1E-15
list_jointness[-1] -= 1E-15

results = []
for name in name_list:
    print name
    results += [pd.read_pickle("./analyses/"+name+".pkl")]

results = pd.concat(results)
plot_results(results, list_jointness)

