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

name = "JointEnvResults"
# name = "IndepEnvResults"

sim_results = pd.read_pickle("./"+name+".pkl")

plot_results(sim_results, figsize=(9, 4.5))


with sns.axes_style('ticks'):
    sns.factorplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']],
          units='Simulation Number', hue='Model', estimator=np.mean,
          palette='Set2')
    sns.despine()
    
    
df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim_results.Model)]
model = []
for m in set(sim_results.Model):
    model += [m] * (sim_results[sim_results.Model == m]['Simulation Number'].max() + 1)
df1 = pd.DataFrame({
        'Cumulative Steps Taken': np.concatenate(cum_steps),
        'Model': model
    })

# sns.set_context('talk')
with sns.axes_style('ticks'):
    fig, ax = plt.subplots(1, 1, figsize=(2, 3))
    cc = sns.color_palette('Set2')
    sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette=cc[1:],
                   order=["Independent", "Joint", 'Hierarchical', 'Meta']
                   )
    ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
    ax.plot([-0.5, 3], [ybar, ybar], 'r--')
    ax.set_ylabel('Total Steps')
    ax.set_xticklabels(['Indep.', 'Joint', 'Hier.', 'Meta'])

    sns.despine()
    plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
    
    
    
df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number', 'Trial Number']).mean()
df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
df0 = df0.rename(index=str, columns={'n actions taken': "Cumulative Steps Taken"})
df0 = df0[df0['Trial Number'] == df0['Trial Number'].max()]
# sim1.groupby(['Model', 'Simulation Number']).mean()
print df0.groupby('Model').mean()['Cumulative Steps Taken']
print df0.groupby('Model')['Cumulative Steps Taken'].std()
