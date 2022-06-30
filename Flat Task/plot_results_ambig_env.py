#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 03:30:52 2020
@author: Rex
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import ttest_ind as ttest


def plot_one_result(name, savefig):
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")

    results_fl = sim_results[sim_results['Model'] == 'Flat']
    results_ic = sim_results[sim_results['Model'] == 'Independent']
    results_jc = sim_results[sim_results['Model'] == 'Joint']
    results_h = sim_results[sim_results['Model'] == 'Hierarchical']
    results_mx = sim_results[sim_results['Model'] == 'Meta']
    
    sim_results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])
    # sim_results = filter_long_trials(sim_results)

    bar_colours = sns.color_palette("Set2")[1:]
    graph_colours = [(0,0,0)] + sns.color_palette("Set2")[1:]
    
    # separate joint and independent contexts
    if name=="AmbigEnvResults_mixed":
        ctx_set1 = sim_results[sim_results['context'] < 6]
        ctx_set2 = sim_results[(sim_results['context'] > 5)]
        tag1, tag2 = 'joint', 'indep'
    elif name=="AmbigEnvResults_cond_ind":
        ctx_set1 = sim_results[sim_results['context'] < 8]
        ctx_set2 = sim_results[(sim_results['context'] > 7)]
        tag1, tag2 = 'set1', 'set2'
    else:
        print("Unrecognised data set")
        raise


    # analyses for all contexts
    sns.set_context('talk')
    with sns.axes_style('ticks'):
        plt.figure(figsize=(5, 4.5))
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.gca().set_yticks(range(0,45,10))
        plt.gca().set_ylim(0,45)
        sns.despine()

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
    if name=="AmbigEnvResults_mixed":
        axins.set_yticks(np.linspace(0.,0.3,4))
        plt.ylim((0.,0.3))
    elif name=="AmbigEnvResults_cond_ind":
        axins.set_yticks(np.linspace(-0.1,0.3,5))
        plt.ylim((-0.1,0.3))
        
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx.png', dpi=300, bbox_inches='tight')
        
    # output test statistics
    if name=="AmbigEnvResults_mixed":
        stat_compare_agents('Joint', goal_data)
    elif name=="AmbigEnvResults_cond_ind":
        stat_compare_agents('Independent', goal_data)
        
    # violin plot
    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette="Set2",
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
        plt.gca().set_ylabel('Total Steps')
        plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        if savefig:
            plt.savefig("figs/"+name+'_violin.png', dpi=300, bbox_inches='tight')



    # analyses for context set 1: joint in mixed case, indep 1 in cond ind case 
    sns.set_context('talk')
    with sns.axes_style('ticks'):
        plt.figure(figsize=(5, 4.5))
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=ctx_set1[ctx_set1['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.gca().set_yticks(range(0,45,10))
        plt.gca().set_ylim(0,45)
        sns.despine()

    goal_data = ctx_set1[ctx_set1['In goal']]
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
    if name=="AmbigEnvResults_mixed":
        axins.set_yticks(np.linspace(0.,0.6,4))
        plt.ylim((0.,0.6))
    elif name=="AmbigEnvResults_cond_ind":
        axins.set_yticks(np.linspace(-0.1,0.3,5))
        plt.ylim((-0.1,0.3))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_'+tag1+'.png', dpi=300, bbox_inches='tight')

    # output test statistic
    stat_compare_agents('Independent', goal_data)
    
    # violin plot
    df0 = ctx_set1[ctx_set1['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(ctx_set1.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette="Set2",
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
        plt.gca().set_ylabel('Total Steps')
        plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        if savefig:
            plt.savefig("figs/"+name+'_violin_'+tag1+'.png', dpi=300, bbox_inches='tight')


    
    # analyses for context set 2: indep in both cases
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=ctx_set2[ctx_set2['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.gca().set_yticks(range(0,45,10))
        plt.gca().set_ylim(0,45)
        sns.despine()

    goal_data = ctx_set2[ctx_set2['In goal']]
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
    if name=="AmbigEnvResults_mixed":
        axins.set_yticks(np.linspace(-0.1,0.3,5))
        plt.ylim((-0.1,0.3))
    elif name=="AmbigEnvResults_cond_ind":
        axins.set_yticks(np.linspace(-0.1,0.3,5))
        plt.ylim((-0.1,0.3))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_'+tag2+'.png', dpi=300, bbox_inches='tight')
        
    # output test statistic
    if name=="AmbigEnvResults_mixed":
        stat_compare_agents('Joint', goal_data)
    elif name=="AmbigEnvResults_cond_ind":
        stat_compare_agents('Independent', goal_data)
    
    # violin plot
    df0 = ctx_set2[ctx_set2['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(ctx_set2.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette="Set2",
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
        plt.gca().set_ylabel('Total Steps')
        plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        if savefig:
            plt.savefig("figs/"+name+'_violin_'+tag2+'.png', dpi=300, bbox_inches='tight')




def filter_long_trials(sim_results):
    results_fl = sim_results[sim_results['Model'] == 'Flat']
    
    # compute cumulative steps to filter out long trials
    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number', 'context']).sum()
    cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim_results.Model)]
    model = []
    for m in set(sim_results.Model):
        model += [m] * (sim_results[sim_results.Model == m]['Simulation Number'].max() + 1)
    df1 = pd.DataFrame({'Cumulative Steps Taken': np.concatenate(cum_steps),'Model': model})

    # filter out trials exceeding threshold length
    threshold = df1[df1['Model']=='Flat']['Cumulative Steps Taken'].mean() + 2.5*df1[df1['Model']=='Flat']['Cumulative Steps Taken'].std()
    n_trials = results_fl['Simulation Number'].max()+1
    df1['Failed'] = df1['Cumulative Steps Taken'] > threshold
    df1['Simulation Number'] = np.concatenate([range(n_trials) for m in set(sim_results.Model)])
    
    failed_set = df1[df1['Failed']]

    new_sim_results = results_fl
    for m in ['Independent', 'Joint', 'Hierarchical', 'Meta']:
        truncated_sims = list(set( failed_set[failed_set['Model']==m]['Simulation Number']))

        tmp = sim_results[sim_results['Model']==m]
        tmp_df = tmp[~tmp['Simulation Number'].isin(truncated_sims)]
        new_sim_results = new_sim_results.append(tmp_df, ignore_index=True)

    return new_sim_results


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
    




        

name_list = ["AmbigEnvResults_mixed", "AmbigEnvResults_cond_ind"]

savefig = False

for name in name_list:
    print name
        
    plot_one_result(name, savefig)
    
    
    