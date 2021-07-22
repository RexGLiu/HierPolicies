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


name_list = ["AmbigEnv3_bottleneck_10000", 
               "AmbigEnv3_bottleneck_15000", "AmbigEnv3_bottleneck_20000", 
               "AmbigEnv3_bottleneck_25000", "AmbigEnv3_bottleneck_30000", 
               "AmbigEnv3_bottleneck_35000", "AmbigEnv3_bottleneck_40000", 
               "AmbigEnv3_bottleneck_45000", "AmbigEnv3_bottleneck_50000", 
               "AmbigEnv3_bottleneck_55000", "AmbigEnv3_bottleneck_60000",
               "AmbigEnv3_bottleneck_65000", "AmbigEnv3_bottleneck_70000",
               "AmbigEnv3_bottleneck_75000", "AmbigEnv3_bottleneck_80000",
               "AmbigEnv3_bottleneck_85000", "AmbigEnv3_bottleneck_90000",
               "AmbigEnv3_bottleneck_95000", "AmbigEnv3_bottleneck_100000"]

models = ['Flat', 'Independent', 'Joint', 'Hierarchical', 'Meta']

nMAPS = range(10000,105000,5000)

subfolder = "bottleneck_env2_original"


# name_list = ["AmbigEnv3_bottleneck_100", 
#               "AmbigEnv3_bottleneck_200", "AmbigEnv3_bottleneck_300", 
#               "AmbigEnv3_bottleneck_400", "AmbigEnv3_bottleneck_500", 
#               "AmbigEnv3_bottleneck_600", "AmbigEnv3_bottleneck_700", 
#               "AmbigEnv3_bottleneck_800", "AmbigEnv3_bottleneck_900", 
#               "AmbigEnv3_bottleneck_1000"]

# models = ['Flat', 'Independent', 'Joint', 'Meta']

# nMAPS = range(100,1000,100)

# subfolder = "bottleneck_env2"

#name_list = ["AmbigEnv3_bottleneck_100", 
#             "AmbigEnv3_bottleneck_200", "AmbigEnv3_bottleneck_300", 
#              "AmbigEnv3_bottleneck_400", "AmbigEnv3_bottleneck_500", 
#              "AmbigEnv3_bottleneck_600", "AmbigEnv3_bottleneck_700", 
#              "AmbigEnv3_bottleneck_800", "AmbigEnv3_bottleneck_900", 
#              "AmbigEnv3_bottleneck_1000", "AmbigEnv3_bottleneck_1100",
#             "AmbigEnv3_bottleneck_1200", "AmbigEnv3_bottleneck_1300",
#             "AmbigEnv3_bottleneck_1400", "AmbigEnv3_bottleneck_1500",
#             "AmbigEnv3_bottleneck_1600", "AmbigEnv3_bottleneck_1700",
#             "AmbigEnv3_bottleneck_1800", "AmbigEnv3_bottleneck_1900",
#             "AmbigEnv3_bottleneck_2000"]
#
#models = ['Flat', 'Independent', 'Joint', 'Meta']
#
#nMAPS = range(100,2100,100)
#
#subfolder = "bottleneck_env2c"

data_series = pd.DataFrame()

for ii, name in enumerate(name_list):
    print name
    
#    sim_results = pd.read_pickle("./analyses/new_data/"+name+".pkl")
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")


    results_fl = sim_results[sim_results['Model'] == 'Flat']
    results_ic = sim_results[sim_results['Model'] == 'Independent']
    results_jc = sim_results[sim_results['Model'] == 'Joint']
    results_h = sim_results[sim_results['Model'] == 'Hierarchical']
    results_mx = sim_results[sim_results['Model'] == 'Meta']
    
    
    sim_results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])
#    sim_results = pd.concat([results_fl,results_ic,results_jc,results_mx])

    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim_results.Model)]
    model = []
    for m in set(sim_results.Model):
        model += [m] * (sim_results[sim_results.Model == m]['Simulation Number'].max() + 1)
    df1 = pd.DataFrame({'Cumulative Steps Taken': np.concatenate(cum_steps),'Model': model})
    
    threshold = df1[df1['Model']=='Flat'].mean() - 2*df1[df1['Model']=='Flat'].std()
    n_trials = len(df1[df1['Model']=='Flat'])

    max_series = df1.groupby('Model').max()['Cumulative Steps Taken']
    mean_series = df1.groupby('Model').mean()['Cumulative Steps Taken']
    std_series = df1.groupby('Model').std()['Cumulative Steps Taken']
    excess_series = df1[df1['Cumulative Steps Taken'] > float(threshold)]

    for m in models:
        data_series = data_series.append({'number of bottleneck hypotheses': nMAPS[ii],
            'mean number of steps': mean_series[m], 
            'max number of steps': max_series[m],
            'std number of steps': std_series[m],
            'frac truncated': len(excess_series[excess_series['Model']== m])/float(n_trials),
            'Model': m}, ignore_index=True)
        
with sns.axes_style('ticks'):
    sns.factorplot(x='number of bottleneck hypotheses', y='mean number of steps', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/"+subfolder+"_mean_vs_bottleneck2.png", dpi=300, bbox_inches='tight')

with sns.axes_style('ticks'):
    sns.factorplot(x='number of bottleneck hypotheses', y='max number of steps', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/"+subfolder+"_max_vs_bottleneck2.png", dpi=300, bbox_inches='tight')

with sns.axes_style('ticks'):
    sns.factorplot(x='number of bottleneck hypotheses', y='std number of steps', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/"+subfolder+"_std_vs_bottleneck2.png", dpi=300, bbox_inches='tight')

data_series = data_series[data_series['Model']!='Flat']
with sns.axes_style('ticks'):
    sns.factorplot(x='number of bottleneck hypotheses', y='frac truncated', data=data_series,
                   hue='Model', palette='Set2')
    sns.despine()
    plt.savefig("figs/"+subfolder+"_frac_trunc_vs_bottleneck2.png", dpi=300, bbox_inches='tight')