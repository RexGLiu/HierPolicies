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

for ii, name in enumerate(name_list):
    print name
    
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
    
    flat_mean = df1[df1['Model']=='Flat'].mean()
    n_trials = len(df1[df1['Model']=='Flat'])

    max_series = df1.groupby('Model').max()['Cumulative Steps Taken']
    mean_series = df1.groupby('Model').mean()['Cumulative Steps Taken']
    std_series = df1.groupby('Model').std()['Cumulative Steps Taken']
    excess_series = df1[df1['Cumulative Steps Taken'] > float(flat_mean)]

    for m in models:
        data_series = data_series.append({'number of MAP hypotheses': (ii+1)*10,
            'mean number of steps': mean_series[m], 
            'max number of steps': max_series[m],
            'std number of steps': std_series[m],
            'frac truncated': len(excess_series[excess_series['Model']== m])/float(n_trials),
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