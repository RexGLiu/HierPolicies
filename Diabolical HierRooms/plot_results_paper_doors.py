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
from scipy.stats import ttest_ind as ttest


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
            
    print('Door seq')
    stat_compare_agents('Independent', X_first_by_upper_context)
    print('')
    
    

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
            
    print('Sublvl')
    stat_compare_agents('Independent', X_first_by_sublvl)
    print('')
    

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
            
    print('Door 0')
    stat_compare_agents('Independent', X_door0)
    print('')
    
    

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
            
    print('Door 1')
    stat_compare_agents('Independent', X_door1)
    print('')
    
    

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
            
    print('Door 2')
    stat_compare_agents('Independent', X_door2)
    print('')
    
    

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
            
    print('Door 3')
    stat_compare_agents('Independent', X_door3)
    print('')
    
    


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
    
    baseline_data = goal_data[goal_data['Model'] == baseline_model]['Reward']
    
    for m in set(goal_data.Model):
        if m == baseline_model:
            continue
        
        model_data = goal_data[goal_data['Model'] == m]['Reward']
        t, p = ttest(model_data, baseline_data, equal_var=False)
        df = WelchSatterthwaitte(model_data, baseline_data)
        print(m, t, p/2, df, np.mean(model_data), np.mean(baseline_data), np.mean(model_data)-np.mean(baseline_data))





savefig=False
fig_size = (5, 4.5)
plot_one_result(savefig)
