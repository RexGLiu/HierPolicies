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


def plot_one_result(name, legend, savefig):
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")

    results_fl = sim_results[sim_results['Model'] == 'Flat']
    results_ic = sim_results[sim_results['Model'] == 'Independent']
    results_jc = sim_results[sim_results['Model'] == 'Joint']
    results_h = sim_results[sim_results['Model'] == 'Hierarchical']
    results_mx = sim_results[sim_results['Model'] == 'Meta']
    
    sim_results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])
    
    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim_results.Model)]
    model = []
    for m in set(sim_results.Model):
        model += [m] * (sim_results[sim_results.Model == m]['Simulation Number'].max() + 1)
    df1 = pd.DataFrame({'Cumulative Steps Taken': np.concatenate(cum_steps),'Model': model})


    # filter out trials exceeding threshold length
    threshold = df1[df1['Model']=='Flat']['Cumulative Steps Taken'].mean() - 2*df1[df1['Model']=='Flat']['Cumulative Steps Taken'].std()
    n_trials = results_fl['Simulation Number'].max()+1
    df1['Failed'] = df1['Cumulative Steps Taken'] > threshold
    df1['Simulation Number'] = np.concatenate([range(n_trials) for m in set(sim_results.Model)])
    
    for m in set(sim_results.Model):
        if m == 'Flat':
            continue
        
        truncated_sims = set( df1[df1['Failed']][df1['Model']==m]['Simulation Number'])
        tmp = sim_results[sim_results['Model']==m]
        sim_results[sim_results['Model']==m] = tmp[~tmp.isin(truncated_sims)]
        
    df1 = df1[(df1.Model=='Flat') | (~df1.Failed)]

    sns.set_context('talk')
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        sns.pointplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']], 
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        if legend:
            plt.legend(prop={'size': 14})
        else:
            plt.gca().get_legend().remove()
        sns.despine()
    plt.tight_layout()
    plt.gca().set_yticks(range(0,50,10))
    plt.gca().set_ylim(0,40)
    if savefig:
        plt.savefig("figs/"+name+"_ctx.png", dpi=300, bbox_inches='tight')


    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
        plt.gca().set_ylabel('Total Steps')
        plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])
        sns.despine()
    plt.tight_layout()
    plt.gca().set_yticks(range(200,700,100))
    plt.gca().set_ylim(200,650)
    if savefig:
        plt.savefig("figs/"+name+"_violin.png", dpi=300, bbox_inches='tight')


filename_list = ["IndepEnvResults", "JointEnvResults"]
n_files = len(filename_list)

savefig = True

for idx, name in enumerate(filename_list):
    print name
    
    if idx==0:
        legend=True
    else:
        legend=False
    
    plot_one_result(name, legend, savefig)
