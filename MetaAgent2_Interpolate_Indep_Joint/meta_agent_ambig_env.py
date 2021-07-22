#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:04:10 2020

@author: rex
"""

import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# custom libraries used
from models.grid_world import Experiment
from models.agents import IndependentClusterAgent, JointClusteringAgent, FlatAgent, MetaAgent, HierarchicalAgent
from models.experiment_designs.experiment1 import gen_task_param
sns.set_context('paper', font_scale=1.5)


n_sims = 5

# alpha is sample from the distribution
# log(alpha) ~ N(alpha_mu, alpha_scale)
alpha_mu = -0.5
alpha_scale = 1.0

beta_mu = 2.0
beta_scale = 0.5

inv_temp = 10.0
goal_prior = 1e-10 
mapping_prior = 1e-10
pruning_threshold = 500.0
evaluate = False

np.random.seed(0)

# pre generate a set of tasks for consistency. 
list_tasks = [gen_task_param() for _ in range(n_sims)]

# pre draw the alphas for consistency
list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale)) 
              for _ in range(n_sims)]

list_beta = [np.exp(scipy.random.normal(loc=beta_mu, scale=beta_scale))
                for _ in range(n_sims)]

def sim_agent(AgentClass, name='None', flat=False, meta=False, hier=False):
    results = []
    print name
    for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

        if not flat:
            if hier:
                jointness = 0.01
                agent_kwargs = dict(alpha=1., jointness=jointness, sample_tau = 1, inv_temp=inv_temp, mapping_prior=mapping_prior,
                                    goal_prior=goal_prior)
            else:
                agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=list_beta[ii], mapping_prior=mapping_prior,
                                goal_prior=goal_prior)
        else:
            agent_kwargs = dict(inv_temp=list_beta[ii], goal_prior=goal_prior, mapping_prior=mapping_prior)
            
        if meta:
            p = np.random.uniform(0.0, 1.00)
            agent_kwargs['mix_biases'] = [np.log(p), np.log(1-p)]
            agent_kwargs['update_new_c_only'] = False

        agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)
        
        _res = None
        while _res is None:
            _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)
        _res[u'Model'] = [name] * len(_res)
        _res[u'Iteration'] = [ii] * len(_res)
        results.append(_res)
    return pd.concat(results)


results_ic = sim_agent(IndependentClusterAgent, name='Independent')
results_jc = sim_agent(JointClusteringAgent, name='Joint')
results_h = sim_agent(HierarchicalAgent, name='Hierarchical', hier=True)
results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
results = pd.concat([results_ic, results_jc, results_h, results_fl, results_meta])





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
    
with sns.axes_style('ticks'):
    g = sns.factorplot(x='Context', y='Reward', data=in_goal[in_goal['Contexts'] == 'Test'], 
                   kind='bar', palette='Set2', col='Model', units='Iteration')
    g.set_xticklabels(['T1', 'T2', 'T3', 'T4'])
    sns.despine(offset=10)
    






#import pandas as pd
#import numpy as np
#import seaborn as sns
#from matplotlib import gridspec
#
#
#def plot_results(df, figsize=(6, 3), sharey=True):
#    with sns.axes_style('ticks'):
#
#        _ = plt.figure(figsize=figsize)
#        gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.4)
#
#        ax0 = plt.subplot(gs[0])
#        ax1 = plt.subplot(gs[1])
#
#        # define the parameters to plot the results
#        # cc = sns.color_palette("Set 2")
#        df0 = df[df['In goal']].groupby(['Model', 'Simulation Number', 'Trial Number']).mean()
#        df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
#        df0 = df0.rename(index=str, columns={'n actions taken': "Cumulative Steps Taken"})
#
#        tsplot_kwargs = dict(
#            time='Trial Number',
#            value='Cumulative Steps Taken',
#            data=df0,
#            unit='Simulation Number',
#            condition='Model',
#            estimator=np.mean,
#            ax=ax0,
#            color="Set2",
#        )
#
#        sns.tsplot(**tsplot_kwargs)
#        df0 = df[df['In goal']].groupby(['Model', 'Simulation Number']).sum()
#        cum_steps = [df0.loc[m]['n actions taken'].values for m in set(df.Model)]
#        model = []
#        for m in set(df.Model):
#            model += [m] * (df[df.Model == m]['Simulation Number'].max() + 1)
#        df1 = pd.DataFrame({
#                'Cumulative Steps Taken': np.concatenate(cum_steps),
#                'Model': model
#            })
#
#        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax1, palette='Set2',
#                       order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
#                       )
#        ax1.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])
#
#        _, ub = ax1.get_ylim()
#        ax1.set_ylim([0, ub])
#        if sharey is True:
#            _, ub = ax0.get_ylim()
#            ax0.set_ylim([0, ub])
#
#
#        sns.despine(offset=5)
#
#
#plot_results(results, figsize=(9, 4.5))
#
#with sns.axes_style('ticks'):
#    sns.factorplot(x='Times Seen Context', y='n actions taken', data=results[results['In goal']],
#          units='Simulation Number', hue='Model', estimator=np.mean,
#          palette='Set2')
#    sns.despine()
#
#df0 = results[results['In goal']].groupby(['Model', 'Simulation Number']).sum()
#cum_steps = [df0.loc[m]['n actions taken'].values for m in set(results.Model)]
#model = []
#for m in set(results.Model):
#    model += [m] * (results[results.Model == m]['Simulation Number'].max() + 1)
#df1 = pd.DataFrame({
#        'Cumulative Steps Taken': np.concatenate(cum_steps),
#        'Model': model
#    })
#
## sns.set_context('talk')
#sns.set_context('paper', font_scale=1.5)
#
#with sns.axes_style('ticks'):
#    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
#    cc = sns.color_palette('Set2')
#    sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette=cc[1:],
#                   order=["Independent", "Joint", 'Hierarchical', 'Meta']
#                   )
#    ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
#    ax.plot([-0.5, 3], [ybar, ybar], 'r--')
#    ax.set_ylabel('Total Steps')
#    ax.set_xticklabels(['Indep.', 'Joint', 'Hier.', 'Meta'])
#    sns.despine()
#    plt.savefig('sim1_mixed.png', dpi=300, bbox_inches='tight')
#    
#    
#
#
#
#df0 = results[results['In goal']].groupby(['Model', 'Simulation Number', 'Trial Number']).mean()
#df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
#df0 = df0.rename(index=str, columns={'n actions taken': "Cumulative Steps Taken"})
#df0 = df0[df0['Trial Number'] == df0['Trial Number'].max()]
#print df0.groupby('Model').mean()['Cumulative Steps Taken']
#print df0.groupby('Model')['Cumulative Steps Taken'].std()