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

def plot_one_result(name, popular, savefig):
    sim_results = pd.read_pickle("./analyses/paper data/"+name+".pkl")

    results_fl = sim_results[sim_results['Model'] == 'Flat']
    results_ic = sim_results[sim_results['Model'] == 'Independent']
    results_jc = sim_results[sim_results['Model'] == 'Joint']
    results_h = sim_results[sim_results['Model'] == 'Hierarchical']
    results_mx = sim_results[sim_results['Model'] == 'Meta']
    
    sim_results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])
    sim_results = filter_long_trials(sim_results)

    bar_colours = sns.color_palette("Set2")[1:]
    graph_colours = [(0,0,0)] + sns.color_palette("Set2")[1:]
    
    # separate joint and independent contexts
    if popular:
        joint_results = sim_results[sim_results['context'] < 4]
        indep_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 34)]
        ambig_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 6)]
        train_results = sim_results[sim_results['context'] < 34]
        test_joint_results = sim_results[sim_results['context'] == 34]
        test_indep_results = sim_results[sim_results['context'] > 34]
        test_indep_results1 = sim_results[sim_results['context'] == 35]
        test_indep_results2 = sim_results[sim_results['context'] == 36]
    else:
        joint_results = sim_results[sim_results['context'] < 4]
        indep_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 16)]
        ambig_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 6)]
        train_results = sim_results[sim_results['context'] < 16]
        test_joint_results = sim_results[sim_results['context'] == 16]
        test_indep_results = sim_results[sim_results['context'] > 16]
        test_indep_results1 = sim_results[sim_results['context'] == 17]
        test_indep_results2 = sim_results[sim_results['context'] == 18]




    sns.set_context('talk')
    with sns.axes_style('ticks'):
        plt.figure(figsize=(5, 4.5))
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
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
    axins.set_yticks(np.linspace(0.4,0.6,5))
    plt.ylim((0.4,0.6))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx.png', dpi=300, bbox_inches='tight')
        
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



    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=train_results[train_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.ylim((0,45))
        plt.gca().get_legend().remove()
        sns.despine()
            
    goal_data = train_results[train_results['In goal']]
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
    axins.set_yticks(np.linspace(0.4,0.6,5))
    plt.ylim((0.4,0.6))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_train.png', dpi=300, bbox_inches='tight')

    df0 = train_results[train_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
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
            plt.savefig("figs/"+name+'_violin_train.png', dpi=300, bbox_inches='tight')            




    sns.set_context('talk')
    with sns.axes_style('ticks'):
        plt.figure(figsize=(5, 4.5))
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=joint_results[joint_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
        sns.despine()

    goal_data = joint_results[joint_results['In goal']]
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
    axins.set_yticks(np.linspace(0.2,0.6,5))
    plt.ylim((0.2,0.6))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_joint.png', dpi=300, bbox_inches='tight')
    
    df0 = joint_results[joint_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(joint_results.Model):
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
            plt.savefig("figs/"+name+'_violin_joint.png', dpi=300, bbox_inches='tight')




    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=ambig_results[ambig_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
        sns.despine()

    goal_data = ambig_results[ambig_results['In goal']]
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
    axins.set_yticks(np.linspace(-0.2,0.6,5))
    plt.ylim((-0.2,0.5))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_ambig.png', dpi=300, bbox_inches='tight')

    
    df0 = ambig_results[ambig_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(ambig_results.Model):
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
            plt.savefig("figs/"+name+'_violin_ambig.png', dpi=300, bbox_inches='tight')



        
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=indep_results[indep_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
        sns.despine()

    goal_data = ambig_results[ambig_results['In goal']]
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
    axins.set_yticks(np.linspace(-0.1,0.4,6))
    plt.ylim((-0.1,0.4))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_indep.png', dpi=300, bbox_inches='tight')

    
    df0 = indep_results[indep_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(indep_results.Model):
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
            plt.savefig("figs/"+name+'_violin_indep.png', dpi=300, bbox_inches='tight')


        
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_joint_results[test_joint_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
        sns.despine()

    goal_data = test_joint_results[test_joint_results['In goal']]
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
    axins.set_yticks(np.linspace(0.3,0.9,4))
    plt.ylim((0.3,0.9))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_test_joint.png', dpi=300, bbox_inches='tight')

    
    df0 = test_joint_results[test_joint_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_joint_results.Model):
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
            plt.savefig("figs/"+name+'_violin_test_joint.png', dpi=300, bbox_inches='tight')



        
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_indep_results[test_indep_results['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
        sns.despine()

    goal_data = test_indep_results[test_indep_results['In goal']]
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
    axins.set_yticks(np.linspace(-0.1,0.3,5))
    plt.ylim((-0.1,0.3))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_test_indep.png', dpi=300, bbox_inches='tight')

    
    df0 = test_indep_results[test_indep_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_indep_results.Model):
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
            plt.savefig("figs/"+name+'_violin_test_indep.png', dpi=300, bbox_inches='tight')


        
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_indep_results1[test_indep_results1['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
        sns.despine()

    goal_data = test_indep_results1[test_indep_results1['In goal']]
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
    axins.set_yticks(np.linspace(-0.2,0.6,5))
    plt.ylim((-0.2,0.6))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_test_indep1.png', dpi=300, bbox_inches='tight')

    
    df0 = test_indep_results1[test_indep_results1['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_indep_results1.Model):
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
            plt.savefig("figs/"+name+'_violin_test_indep1.png', dpi=300, bbox_inches='tight')


        
    plt.figure(figsize=(5, 4.5))
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_indep_results2[test_indep_results2['In goal']],
                        hue_order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette=graph_colours)
        plt.gca().get_legend().remove()
        plt.ylim((0,45))
        sns.despine()

    goal_data = test_indep_results2[test_indep_results2['In goal']]
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
    axins.set_yticks(np.linspace(-0.2,0.6,5))
    plt.ylim((-0.2,0.6))
    sns.despine()
    if savefig:
        plt.savefig("figs/"+name+'_ctx_test_indep2.png', dpi=300, bbox_inches='tight')
    
    df0 = test_indep_results2[test_indep_results2['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_indep_results2.Model):
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
            plt.savefig("figs/"+name+'_violin_test_indep2.png', dpi=300, bbox_inches='tight')



def filter_long_trials(sim_results):
    results_fl = sim_results[sim_results['Model'] == 'Flat']
    
    # compute cumulative steps to filter out long trials
    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim_results.Model)]
    model = []
    for m in set(sim_results.Model):
        model += [m] * (sim_results[sim_results.Model == m]['Simulation Number'].max() + 1)
    df1 = pd.DataFrame({'Cumulative Steps Taken': np.concatenate(cum_steps),'Model': model})

    # filter out trials exceeding threshold length
    threshold = df1[df1['Model']=='Flat']['Cumulative Steps Taken'].mean() - 3*df1[df1['Model']=='Flat']['Cumulative Steps Taken'].std()
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



        

name_list = ["AmbigEnvResults_rare_goal"]
savefig = True


for name in name_list:
    print name
    
    if name == "AmbigEnvResults_popular_goal":
        popular = True
    elif name == "AmbigEnvResults_rare_goal" :
        popular = False
    else:
        print('Unrecognised data file')
        raise
    
    plot_one_result(name, popular, savefig)
    
    
    