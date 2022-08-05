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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_ind as ttest


def plot_one_result(name, savefig, baseline):
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


    # # filter out trials exceeding threshold length
    # threshold = df1[df1['Model']=='Flat']['Cumulative Steps Taken'].mean() + 2.5*df1[df1['Model']=='Flat']['Cumulative Steps Taken'].std()
    # n_trials = results_fl['Simulation Number'].max()+1
    # df1['Failed'] = df1['Cumulative Steps Taken'] > threshold
    # df1['Simulation Number'] = np.concatenate([range(n_trials) for m in set(sim_results.Model)])
    
    # failed_set = df1[df1['Failed']]

    # new_sim_results = results_fl
    # for m in set(sim_results.Model):
    #     if m == 'Flat':
    #         continue
    #     truncated_sims = list(set( failed_set[failed_set['Model']==m]['Simulation Number']))
        
    #     tmp = sim_results[sim_results['Model']==m]
    #     tmp_df = tmp[~tmp['Simulation Number'].isin(truncated_sims)]
    #     new_sim_results = new_sim_results.append(tmp_df, ignore_index=True)

    # sim_results = new_sim_results
        
    # df1 = df1[(df1.Model=='Flat') | (~df1.Failed)]


    bar_colours = sns.color_palette("Set2")[1:]
    graph_colours = [(0,0,0)] + sns.color_palette("Set2")[1:]

    sns.set_context('talk')
    plt.figure(figsize=(5.5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']], 
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        sns.despine()
    plt.tight_layout()
    plt.gca().set_yticks(range(0,50,10))
    plt.gca().set_ylim(0,45)
    

    goal_data = sim_results[sim_results['In goal']]
    goal_data = goal_data[goal_data['Times Seen Context'] == 1]
    action_means = goal_data.groupby(['Times Seen Context', 'Model'])['n actions taken'].mean().reset_index()
    flat_means = float(action_means[action_means['Model']=='Flat']['n actions taken'])

    goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken'] = flat_means - goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken']
    goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken'] = goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken'] / flat_means

    goal_data = goal_data[goal_data['Model'] != 'Flat']
    goal_data = goal_data.rename(columns = {'n actions taken': 'frac improvement'})
    
    axins = inset_axes(ax,  "50%", "40%" ,loc="upper right", borderpad=0)
    sns.barplot(x='Model', y='frac improvement', data=goal_data,
                        order=["Independent", "Joint", "Hierarchical", "Meta"],
                        estimator=np.mean, palette=bar_colours, ax=axins)
    axins.set(xticklabels=["Indep", "Joint", "Hier", "Meta"])
    axins.set_xlabel("")
    axins.set_ylabel("frac improvement", fontsize=13)
    axins.tick_params(axis='both', which='major', labelsize=13)
    
    if name == 'IndepEnvResults':
        axins.set_yticks(np.linspace(-0.1,0.2,4))
        plt.ylim((-0.1,0.2))
    elif name == 'JointEnvResults':
        axins.set_yticks(np.linspace(0.0,0.4,5))
        plt.ylim((0.,0.4))

    # axins.set_yticks(np.linspace(0.0,0.5,6))
    # plt.ylim((0.0,0.5))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+"_ctx.png", dpi=300, bbox_inches='tight')
        
    # output test statistics
    if name == 'IndepEnvResults':
        stat_compare_agents('Joint', goal_data)
    elif name == 'JointEnvResults':
        stat_compare_agents('Independent', goal_data)
    else:
        print('Unrecognised dataset')
        raise
        


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
    # plt.gca().set_yticks(range(200,700,100))
    # plt.gca().set_ylim(200,650)
    if savefig:
        plt.savefig("figs/"+name+"_violin.png", dpi=300, bbox_inches='tight')


def WelchSatterthwaitte(dataset1, dataset2):
    N1 = dataset1.size
    N2 = dataset2.size
    nu1 = N1-1
    nu2 = N2-1

    s1 = dataset1.std()
    s2 = dataset2.std()
    
    
    S1 = s1*s1/N1
    S2 = s2*s2/N2
    
    nu = (S1 + S2)*(S1 + S2)/(S1*S1/nu1 + S2*S2/nu2)
    return nu


def stat_compare_agents(baseline_model, goal_data):
    # runs statistical test comparing each agent's performance against a baseline agent
    
    baseline_data = goal_data[goal_data['Model'] == baseline_model]['frac improvement']
    
    for m in set(goal_data.Model):
        if m == baseline_model:
            continue
        
        model_data = goal_data[goal_data['Model'] == m]['frac improvement']
        t, p = ttest(model_data, baseline_data, equal_var=False)
        df = WelchSatterthwaitte(model_data, baseline_data)
        print(m, t, p/2, df, np.mean(model_data), np.mean(baseline_data), np.mean(model_data)-np.mean(baseline_data))
    



filename_list = ["IndepEnvResults", "JointEnvResults"]
baseline = ['Independent', 'Joint']


n_files = len(filename_list)

savefig = False

for idx, name in enumerate(filename_list):
    print name
    
    plot_one_result(name, savefig, baseline)
