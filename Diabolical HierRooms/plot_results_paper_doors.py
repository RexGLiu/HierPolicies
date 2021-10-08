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


def plot_one_result(savefig):
    results = pd.read_pickle("./analyses/HierarchicalRooms_indep.pkl")

    times_ctx = 1
    X_goals = results[results['In Goal']]
    X_upper = X_goals[X_goals['Room'] % 4 == 0]
    
    X_first_by_room = X_goals[X_goals['Times Seen Context'] == times_ctx]
    X_first_by_upper_context = X_upper[X_upper['Times seen door context'] == times_ctx]
    X_first_by_sublvl = X_first_by_room[X_first_by_room['Room'] % 4 != 0]

    sns.set_context('talk')
    
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        sns.barplot(data=X_first_by_upper_context, x='Model', y='Reward', 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
        ax.set_ylabel('Proportion of successful first visits')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
        # ax.set_title('Doors')
        ax.set_ylim([0,1])
        plt.tight_layout()
        
        if savefig:
            plt.savefig('figs/HierRooms_indep_prop_successful_first_visits_doors.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        sns.barplot(data=X_first_by_sublvl, x='Model', y='Reward', 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
        ax.set_ylabel('Proportion of successful first visits')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
        # ax.set_title('Sublvl goals')
        ax.set_ylim([0,1])
        plt.tight_layout()

        if savefig:
            plt.savefig('figs/HierRooms_indep_prop_successful_first_visits_sublvls.png', dpi=300, bbox_inches='tight')



    # Successful first visit by individual door
    X_door0 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 0]
    X_door1 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 1]
    X_door2 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 2]
    X_door3 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 3]


    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        sns.barplot(data=X_door0, x='Model', y='Reward', 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
        ax.set_ylabel('Proportion of successful first visits')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
        # ax.set_title('First door')
        ax.set_ylim([0,1])
        plt.tight_layout()

        if savefig:
            plt.savefig('figs/HierRooms_indep_prop_successful_first_visits_door0.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        sns.barplot(data=X_door1, x='Model', y='Reward', 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
        ax.set_ylabel('Proportion of successful first visits')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
        # ax.set_title('Second door')
        ax.set_ylim([0,1])
        plt.tight_layout()

        if savefig:
            plt.savefig('figs/HierRooms_indep_prop_successful_first_visits_door1.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        sns.barplot(data=X_door2, x='Model', y='Reward', 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
        ax.set_ylabel('Proportion of successful first visits')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
        # ax.set_title('Third door')
        ax.set_ylim([0,1])
        plt.tight_layout()

        if savefig:
            plt.savefig('figs/HierRooms_indep_prop_successful_first_visits_door2.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=fig_size)
    ax = plt.gca()
    with sns.axes_style('ticks'):
        sns.barplot(data=X_door3, x='Model', y='Reward', 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
        ax.set_ylabel('Proportion of successful first visits')
        ax.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
        # ax.set_title('Fourth door')
        ax.set_ylim([0,1])
        plt.tight_layout()

        if savefig:
            plt.savefig('figs/HierRooms_indep_prop_successful_first_visits_door3.png', dpi=300, bbox_inches='tight')


    





savefig=True
fig_size = (5, 4.5)
plot_one_result(savefig)
