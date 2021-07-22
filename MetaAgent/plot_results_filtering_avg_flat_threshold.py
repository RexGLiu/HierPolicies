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
from scipy.stats import kurtosis


name_list = ["AmbigEnv3_particles10", "AmbigEnv3_particles20",
             "AmbigEnv3_particles30", "AmbigEnv3_particles40",
            "AmbigEnv3_particles50", "AmbigEnv3_particles60",
            "AmbigEnv3_particles70", "AmbigEnv3_particles80",
            "AmbigEnv3_particles90", "AmbigEnv3_particles100"]


models = ['Flat', 'Independent', 'Joint', 'Hierarchical', 'Meta']
data_series = pd.DataFrame()

trunc_series = dict()

for ii, name in enumerate(name_list):
    print name
    
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")

    results_fl = sim_results[sim_results['Model'] == 'Flat']
    results_ic = sim_results[sim_results['Model'] == 'Independent']
    results_jc = sim_results[sim_results['Model'] == 'Joint']
    results_h = sim_results[sim_results['Model'] == 'Hierarchical']
    results_mx = sim_results[sim_results['Model'] == 'Meta']
    
    threshold =  results_fl['n actions taken'].max()
    n_trials = max(results_fl['Simulation Number'])+1
    trunc_series['Flat'] = 0
    
    truncated_sims = set(results_ic['Simulation Number'][results_ic['n actions taken'] > threshold])    
    results_ic = results_ic[~results_ic['Simulation Number'].isin(truncated_sims)]
    trunc_series['Independent'] = 1-len(set(results_ic['Simulation Number']))/float(n_trials)
    
    truncated_sims = set(results_jc['Simulation Number'][results_jc['n actions taken'] > threshold])    
    results_jc = results_jc[~results_jc['Simulation Number'].isin(truncated_sims)]
    trunc_series['Joint'] = 1-len(set(results_jc['Simulation Number']))/float(n_trials)
    
    truncated_sims = set(results_h['Simulation Number'][results_h['n actions taken'] > threshold])    
    results_h = results_h[~results_h['Simulation Number'].isin(truncated_sims)]
    trunc_series['Hierarchical'] = 1-len(set(results_h['Simulation Number']))/float(n_trials)
    
    truncated_sims = set(results_mx['Simulation Number'][results_mx['n actions taken'] > threshold])    
    results_mx = results_mx[~results_mx['Simulation Number'].isin(truncated_sims)]
    trunc_series['Meta'] = 1-len(set(results_mx['Simulation Number']))/float(n_trials)
    
    sim_results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])

    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    cum_steps = [df0.loc[m]['n actions taken'].values for m in models]

    for jj, m in enumerate(models):
        data_series = data_series.append({'number of MAP hypotheses': (ii+1)*10,
            'mean number of steps': np.mean(cum_steps[jj]), 
            'max number of steps': max(cum_steps[jj]),
            'std number of steps': np.std(cum_steps[jj]),
            'frac truncated': trunc_series[m],
            'Model': m}, ignore_index=True)

nMAPS = range(10,110,10)

with sns.axes_style('ticks'):
    sns.factorplot(x='number of MAP hypotheses', y='mean number of steps', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/mean_vs_n_MAPs.png", dpi=300, bbox_inches='tight')

with sns.axes_style('ticks'):
    sns.factorplot(x='number of MAP hypotheses', y='max number of steps', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/max_vs_n_MAPs.png", dpi=300, bbox_inches='tight')

with sns.axes_style('ticks'):
    sns.factorplot(x='number of MAP hypotheses', y='std number of steps', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/std_vs_n_MAPs.png", dpi=300, bbox_inches='tight')

data_series = data_series[data_series['Model']!='Flat']
with sns.axes_style('ticks'):
    sns.factorplot(x='number of MAP hypotheses', y='frac truncated', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/frac_trunc_vs_n_MAPS.png", dpi=300, bbox_inches='tight')