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


def plot_one_result(name):
    results = pd.read_pickle("./analyses/"+name+".pkl")
    
    results_fl = results[results['Model'] == 'Flat']
    results_ic = results[results['Model'] == 'Independent']
    results_jc = results[results['Model'] == 'Joint']
    results_h = results[results['Model'] == 'Hierarchical']
    results_mx = results[results['Model'] == 'Meta']
    
    results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])
    
    # with sns.axes_style('ticks'):
    #     sns.factorplot(x='Times Seen Context', y='Steps Taken', data=results[results['In Goal']],
    #                     units='Iteration', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     plt.savefig("figs/"+name+'_ctx.png', dpi=300, bbox_inches='tight')
        

    # df0 = results[results['In Goal']].groupby(['Model', 'Iteration']).sum()
    # cum_steps = [df0.loc[m]['Steps Taken'].values for m in set(results.Model)]
    # model = []
    # for m in set(results.Model):
    #     model += [m] * (results[results.Model == m]['Iteration'].max() + 1)
    # df1 = pd.DataFrame({'Cumulative Steps Taken': np.concatenate(cum_steps),'Model': model})

    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
    #     ax.plot([-0.5, 4], [ybar, ybar], 'r--')
    #     ax.set_ylabel('Total Steps')
    #     ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.savefig("figs/"+name+'_violin.png', dpi=300, bbox_inches='tight')
    
    
    df0 = results[results['In Goal']].groupby(['Model', 'Iteration', 'Trial Number']).mean()
    df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
    df0 = df0.rename(index=str, columns={'Steps Taken': "Cumulative Steps Taken"})
    df0 = df0[df0['Trial Number'] == df0['Trial Number'].max()]
    # sim1.groupby(['Model', 'Iteration']).mean()
    print df0.groupby('Model').mean()['Cumulative Steps Taken']
    print df0.groupby('Model')['Cumulative Steps Taken'].std()



    in_goal = results[results['In Goal'] ].copy()
    in_goal['Contexts'] = [None] * len(in_goal)
    in_goal.loc[in_goal.Context < 5, 'Contexts'] = 'Training'
    in_goal.loc[in_goal.Context >= 5, 'Contexts'] = 'Test'

    with sns.axes_style('white'):
        g = sns.factorplot(y='Reward', data=in_goal, x='Contexts', 
                    hue='Model', units='Iteration', kind='bar', 
                    estimator=np.mean, palette='Accent', size=4)
        sns.despine(offset=5, trim=False)
        ax = g.axes[0][0]
        ax.set_ylabel('Average reward per trial')
        plt.savefig("figs/"+name+'_rewards.png', dpi=300, bbox_inches='tight')
    
    with sns.axes_style('ticks'):
        g = sns.factorplot(x='Context', y='Reward', data=in_goal[in_goal['Contexts'] == 'Test'], 
                    kind='bar', palette='Set2', col='Model', units='Iteration')
        g.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
        sns.despine(offset=10)
        plt.savefig("figs/"+name+'_T.png', dpi=300, bbox_inches='tight')
    

name_list = ["AmbigEnv2Results_300","AmbigEnv2Results_0_8alpha0_300"]


for name in name_list:
    print name
    
    plot_one_result(name)

