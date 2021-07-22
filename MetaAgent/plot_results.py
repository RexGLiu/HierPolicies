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
from model import plot_results


def plot_one_result(name):
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

    # plot_results(sim_results, figsize=(11, 4.5), name='figs/'+name+'.png')



    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx.png', dpi=300, bbox_inches='tight')

    
    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4.5], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin.png', dpi=300, bbox_inches='tight')
        
# name_list = ["IndepEnvResults_300", "JointEnvResults_300"]
# name_list = ["AmbigEnvResults_5", "AmbigEnvResults_6", "AmbigEnvResults_7",
# "AmbigEnvResults_8", "AmbigEnvResults_9", "AmbigEnvResults_10"]
#name_list = ["IndepEnvResults_alpha_1e10", "JointEnvResults_alpha_1e10", "IndepEnvResults_300_alpha0",
#             "JointEnvResults_300_alpha0", "IndepEnvResults_j", "JointEnvResults_j",
#             "IndepEnvResults_j_opt", "JointEnvResults_j_opt"]
# name_list = ["AmbigEnv2Results_5", "AmbigEnv2Results_6", "AmbigEnv2Results_7",
#              "AmbigEnv2Results_8", "AmbigEnv2Results_9"]
# name_list = ["IndepEnvResults_j", "JointEnvResults_j",
#             "IndepEnvResults_j_opt", "JointEnvResults_j_opt"]
# name_list = ["AmbigEnv3_particles10", "AmbigEnv3_particles20",
#              "AmbigEnv3_particles30", "AmbigEnv3_particles40",
#             "AmbigEnv3_particles50", "AmbigEnv3_particles60",
#             "AmbigEnv3_particles70", "AmbigEnv3_particles80",
#             "AmbigEnv3_particles90", "AmbigEnv3_particles100"]
# name_list = ["AmbigEnv3Results_100max"]
# name_list = ["AmbigEnv3_bottleneck_10000", "AmbigEnv3_bottleneck_100000"]
# name_list = ["AmbigEnv4Results"]
name_list = ["IndepEnvResults_300_fixed_Meta", "JointEnvResults_300_fixed_Meta"]


for name in name_list:
    print name
    
    plot_one_result(name)

