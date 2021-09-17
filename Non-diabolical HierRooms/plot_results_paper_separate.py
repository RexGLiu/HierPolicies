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


def plot_one_result(sim_results, name, savefigs=False, legend=False):
    sim_results = sim_results[sim_results['Reward Collected']==1]

    plt.figure(figsize=fig_size)
    with sns.axes_style('ticks'):
        sns.pointplot(x='Times Seen Context', y='n actions taken in room', data=sim_results[sim_results['Reward Collected']==1],
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette='Set2')
        plt.gca().set_ylim(0,170)
        plt.gca().set_yticks(range(0,170,40))
        if legend:
            plt.legend(prop={'size': 14})
        else:
            plt.gca().get_legend().remove()

        sns.despine()
        plt.tight_layout()
        if savefigs:
            plt.savefig("figs/"+name+'_ctx.png', dpi=300, bbox_inches='tight')


    df0 = sim_results[sim_results['Reward Collected']==1].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken in room'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)


    plt.figure(figsize=fig_size)
    sns.set_context('talk')
    with sns.axes_style('ticks'):
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
                    order=["Flat", "Independent", "Hierarchical"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        plt.plot([-0.5, 2.5], [ybar, ybar], 'r--')
        plt.gca().set_ylabel('Total steps')
        plt.gca().set_xticklabels(['Flat', 'Indep.', 'Hier.'])

        sns.despine()
        plt.tight_layout()
        if savefigs:
            plt.savefig("figs/"+name+'_violin.png', dpi=300, bbox_inches='tight')


    # plots across upper levels only
    plt.figure(figsize=fig_size)
    with sns.axes_style('ticks'):
        sns.pointplot(x='Times Seen Context', y='n actions taken in upper room', data=sim_results[sim_results['Reward Collected']==1],
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette='Set2')
        plt.gca().set_ylim(0,100)
        sns.despine()
        plt.gca().get_legend().remove()
        plt.tight_layout()
        if savefigs:
            plt.savefig("figs/"+name+'_upper_rooms_ctx.png', dpi=300, bbox_inches='tight')


    df0 = sim_results[sim_results['Reward Collected']==1].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken in upper room'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    plt.figure(figsize=fig_size)
    sns.set_context('talk')
    with sns.axes_style('ticks'):
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
                    order=["Flat", "Independent", "Hierarchical"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        plt.plot([-0.5, 2.5], [ybar, ybar], 'r--')
        plt.gca().set_ylabel('Total steps across upper levels')
        plt.gca().set_xticklabels(['Flat', 'Indep.', 'Hier.'])

        sns.despine()
        plt.tight_layout()
        if savefigs:
            plt.savefig("figs/"+name+'_upper_rooms_violin.png', dpi=300, bbox_inches='tight')


    # plots across sublevels only
    plt.figure(figsize=fig_size)
    with sns.axes_style('ticks'):
        sns.pointplot(x='Times Seen Context', y='n actions taken in sublvls', data=sim_results[sim_results['Reward Collected']==1],
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette='Set2')
        plt.gca().set_ylim(0,100)
        sns.despine()
        plt.gca().get_legend().remove()
        plt.tight_layout()
        if savefigs:
            plt.savefig("figs/"+name+'_sublvls_ctx.png', dpi=300, bbox_inches='tight')

    df0 = sim_results[sim_results['Reward Collected']==1].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken in sublvls'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    plt.figure(figsize=fig_size)
    sns.set_context('talk')
    with sns.axes_style('ticks'):
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
                    order=["Flat", "Independent", "Hierarchical"]
                    )
        ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
        plt.plot([-0.5, 2.5], [ybar, ybar], 'r--')
        plt.gca().set_ylabel('Total steps across sublevels')
        plt.gca().set_xticklabels(['Flat', 'Indep.', 'Hier.'])

        sns.despine()
        if savefigs:
            plt.savefig("figs/"+name+'_sublvls_violin.png', dpi=300, bbox_inches='tight')



name_list = ["HierarchicalRooms_indep", "HierarchicalRooms_joint", "HierarchicalRooms_ambig"]

fig_size = (5, 4.5)

for name in name_list:
    print name
    savefigs=False
    
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")
    sim_results = sim_results[sim_results['In goal']]
    
    if name == "HierarchicalRooms_ambig":
        # separate joint, mixed, and independent contexts
        joint_results = sim_results[sim_results['context'] < 9]
        mixed_results = sim_results[(sim_results['context'] > 8) == (sim_results['context'] < 21)]
        indep_results = sim_results[(sim_results['context'] > 20)]

        plot_one_result(sim_results, name+"_full", savefigs, True)
        plot_one_result(joint_results, name+"_joint", savefigs)
        plot_one_result(indep_results, name+"_indep", savefigs)
        plot_one_result(mixed_results, name+"_mixed", savefigs)

    else:    
        if name == "HierarchicalRooms_indep":
            legend=True
        else:
            legend=False
        plot_one_result(sim_results, name, savefigs, legend)

