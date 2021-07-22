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
# from model import plot_results


def plot_one_result(sim_results, name, savefigs=False):
    sim_results = sim_results[sim_results['Reward Collected']==1]

    # plots across all rooms and sublvls
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken in room', data=sim_results[sim_results['Reward Collected']==1],
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette='Set2')
        sns.despine()
        if savefigs:
            plt.savefig("figs/"+name+'_ctx.png', dpi=300, bbox_inches='tight')

    df0 = sim_results[sim_results['Reward Collected']==1].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken in room'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Hierarchical"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 2.5], [ybar, ybar], 'r--')
        ax.set_ylabel('Total steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

        sns.despine()
        if savefigs:
            plt.savefig("figs/"+name+'_violin.png', dpi=300, bbox_inches='tight')
        
        
    # plots across upper rooms only
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken in upper room', data=sim_results[sim_results['Reward Collected']==1],
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette='Set2')
        sns.despine()
        if savefigs:
            plt.savefig("figs/"+name+'_upper_rooms_ctx.png', dpi=300, bbox_inches='tight')

    df0 = sim_results[sim_results['Reward Collected']==1].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken in upper room'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Hierarchical"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 2.5], [ybar, ybar], 'r--')
        ax.set_ylabel('Total steps across upper rooms')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

        sns.despine()
        if savefigs:
            plt.savefig("figs/"+name+'_upper_rooms_violin.png', dpi=300, bbox_inches='tight')


    # plots across sublevels only
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken in sublvls', data=sim_results[sim_results['Reward Collected']==1],
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette='Set2')
        sns.despine()
        if savefigs:
            plt.savefig("figs/"+name+'_sublvls_ctx.png', dpi=300, bbox_inches='tight')

    df0 = sim_results[sim_results['Reward Collected']==1].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken in sublvls'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Hierarchical"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 2.5], [ybar, ybar], 'r--')
        ax.set_ylabel('Total steps across sublevels')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

        sns.despine()
        if savefigs:
            plt.savefig("figs/"+name+'_sublvls_violin.png', dpi=300, bbox_inches='tight')


    
        

name_list = ["HierarchicalRooms_ambig_3"]
# name_list = ["HierarchicalRooms_indep_2"]
# name_list = ["HierarchicalRooms_joint"]
# name_list = ["HierarchicalRooms_joint_3"]
# name_list = ["HierarchicalRooms_indep", "HierarchicalRooms_joint"]
# name_list = ["HierarchicalRooms_ambig_indep_envs_only", "HierarchicalRooms_upper_indep_lower_j"]


for name in name_list:
    print name
    savefigs=True
    
    
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")
    sim_results = sim_results[sim_results['In goal']]
    
    # plot_one_result(sim_results, name, savefigs)

    # separate joint, mixed, and independent contexts
    joint_results = sim_results[sim_results['context'] < 12]
    mixed_results = sim_results[(sim_results['context'] > 11) == (sim_results['context'] < 28)]
    indep_results = sim_results[(sim_results['context'] > 27)]

    plot_one_result(sim_results, name+"_full", savefigs)
    plot_one_result(joint_results, name+"_joint", savefigs)
    plot_one_result(indep_results, name+"_indep", savefigs)
    plot_one_result(mixed_results, name+"_mixed", savefigs)


