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


bar_colours = sns.color_palette("Set2")[1:]
graph_colours = [(0,0,0)] + sns.color_palette("Set2")[1:]

def plot_one_result(sim_results, name, inset_yticks, savefigs=False):
    sim_results = sim_results[sim_results['Reward Collected']==1]

    sns.set_context('talk')
    plt.figure(figsize=fig_size)
    with sns.axes_style('ticks'):
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken in room', data=sim_results,
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette=graph_colours)
        plt.gca().set_ylim(0,170)
        plt.gca().set_yticks(range(0,170,40))
        plt.gca().get_legend().remove()

        plt.tight_layout()

        goal_data = sim_results[sim_results['Times Seen Context']==1]
        action_means = goal_data.groupby(['Times Seen Context', 'Model'])['n actions taken in room'].mean().reset_index()
        flat_means = float(action_means[action_means['Model']=='Flat']['n actions taken in room'])

        goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in room'] = flat_means - goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in room']
        goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in room'] = goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in room'] / flat_means
        
        goal_data = goal_data[goal_data['Model'] != 'Flat']
        goal_data = goal_data.rename(columns = {'n actions taken in room': 'frac improvement'})
    
        axins = inset_axes(ax,  "30%", "40%" ,loc="upper right", borderpad=0)
        sns.barplot(x='Model', y='frac improvement', data=goal_data, order = ["Independent", "Hierarchical"],
                        estimator=np.mean, palette=bar_colours, ax=axins)
        axins.set(xticklabels=["Indep", "Hier"])
        axins.set_xlabel("")
        axins.set_ylabel("frac improvement", fontsize=13)
        axins.tick_params(axis='both', which='major', labelsize=13)
        
        yticks = inset_yticks['all']
        axins.set_yticks(yticks)
        plt.ylim((yticks[0],yticks[-1]))
        
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


    plt.figure(figsize=fig_size)
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
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken in upper room', data=sim_results,
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette=graph_colours)
        plt.gca().set_ylim(0,100)
        plt.gca().get_legend().remove()
        plt.tight_layout()

        goal_data = sim_results[sim_results['Times Seen Context']==1]
        action_means = goal_data.groupby(['Times Seen Context', 'Model'])['n actions taken in upper room'].mean().reset_index()
        flat_means = float(action_means[action_means['Model']=='Flat']['n actions taken in upper room'])

        goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in upper room'] = flat_means - goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in upper room']
        goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in upper room'] = goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in upper room'] / flat_means
        
        goal_data = goal_data[goal_data['Model'] != 'Flat']
        goal_data = goal_data.rename(columns = {'n actions taken in upper room': 'frac improvement'})
    
        axins = inset_axes(ax,  "30%", "40%" ,loc="upper right", borderpad=0)
        sns.barplot(x='Model', y='frac improvement', data=goal_data, order = ["Independent", "Hierarchical"],
                        estimator=np.mean, palette=bar_colours, ax=axins)
        axins.set(xticklabels=["Indep", "Hier"])
        axins.set_xlabel("")
        axins.set_ylabel("frac improvement", fontsize=13)
        axins.tick_params(axis='both', which='major', labelsize=13)
        
        yticks = inset_yticks['upper']
        axins.set_yticks(yticks)
        plt.ylim((yticks[0],yticks[-1]))
        
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
        ax = sns.pointplot(x='Times Seen Context', y='n actions taken in sublvls', data=sim_results,
                        units='Simulation Number', hue='Model', hue_order=["Flat", "Independent", "Hierarchical"], estimator=np.mean,
                        palette=graph_colours)
        plt.gca().set_ylim(0,100)
        plt.gca().get_legend().remove()
        plt.tight_layout()

        goal_data = sim_results[sim_results['Times Seen Context']==1]
        action_means = goal_data.groupby(['Times Seen Context', 'Model'])['n actions taken in sublvls'].mean().reset_index()
        flat_means = float(action_means[action_means['Model']=='Flat']['n actions taken in sublvls'])

        goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in sublvls'] = flat_means - goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in sublvls']
        goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in sublvls'] = goal_data.loc[ goal_data['Times Seen Context']==1, 'n actions taken in sublvls'] / flat_means
        
        goal_data = goal_data[goal_data['Model'] != 'Flat']
        goal_data = goal_data.rename(columns = {'n actions taken in sublvls': 'frac improvement'})
    
        axins = inset_axes(ax,  "30%", "40%" ,loc="upper right", borderpad=0)
        sns.barplot(x='Model', y='frac improvement', data=goal_data, order = ["Independent", "Hierarchical"],
                        estimator=np.mean, palette=bar_colours, ax=axins)
        axins.set(xticklabels=["Indep", "Hier"])
        axins.set_xlabel("")
        axins.set_ylabel("frac improvement", fontsize=13)
        axins.tick_params(axis='both', which='major', labelsize=13)
        
        yticks = inset_yticks['sublvl']
        axins.set_yticks(yticks)
        plt.ylim((yticks[0],yticks[-1]))
        
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
    savefigs=True
    
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")
    sim_results = sim_results[sim_results['In goal']]
    
    if name == "HierarchicalRooms_ambig":
        # separate joint, mixed, and independent contexts
        joint_results = sim_results[sim_results['context'] < 9]
        mixed_results = sim_results[(sim_results['context'] > 8) == (sim_results['context'] < 21)]
        indep_results = sim_results[(sim_results['context'] > 20)]

        inset_yticks = {'all': np.linspace(0.4,0.55,4), 'upper': np.linspace(0.3,0.6,4), 'sublvl': np.linspace(0.45,0.55,3)}
        plot_one_result(sim_results, name+"_full", inset_yticks, savefigs)

        inset_yticks = {'all': np.linspace(0.25,0.55,4), 'upper': np.linspace(0.2,0.6,3), 'sublvl': np.linspace(0.35,0.65,4)}
        plot_one_result(joint_results, name+"_joint", inset_yticks, savefigs)

        inset_yticks = {'all': np.linspace(0.4,0.55,4), 'upper': np.linspace(0.35,0.55,3), 'sublvl': np.linspace(0.45,0.55,3)}
        plot_one_result(indep_results, name+"_indep", inset_yticks, savefigs)

        inset_yticks = {'all': np.linspace(0.4,0.6,3), 'upper': np.linspace(0.35,0.65,4), 'sublvl': np.linspace(0.4,0.6,3)}
        plot_one_result(mixed_results, name+"_mixed", inset_yticks, savefigs)

    else:
        if name == "HierarchicalRooms_indep":
            inset_yticks = {'all': np.linspace(0.5,0.6,3), 'upper': np.linspace(0.45,0.6,4), 'sublvl': np.linspace(0.55,0.65,3)}
        else:
            inset_yticks = {'all': np.linspace(0.4,0.6,3), 'upper': np.linspace(0.4,0.6,3), 'sublvl': np.linspace(0.4,0.6,3)}
        plot_one_result(sim_results, name, inset_yticks, savefigs)

