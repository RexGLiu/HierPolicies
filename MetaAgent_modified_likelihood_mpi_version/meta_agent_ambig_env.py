#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:19:14 2020

@author: rex
"""


from model import plot_results
from model import simulate_mixed_task as simulate_task

# define all of the task parameters
grid_world_size = (6, 6)

mapping_definitions = {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {0: u'right', 1: u'down', 4: u'up', 5: u'left'},
    3: {2: u'up', 3: u'left', 7: u'down', 6: u'right'},
    4: {0: u'up', 3: u'down', 4: u'left', 7: u'right'},
    5: {1: u'right', 2: u'left', 5: u'up', 6: u'down'},
    6: {7: u'right', 4: u'left', 3: u'up', 0: u'down'},
}

# define goal locations
goal_locations = {
    0:(0, 0),
    1:(0, 5),
    2:(5, 0),
    3:(5, 5),
    # 4:(0, 3),
    # 5:(5, 3),
}

# define the mappings for each context, where the ith mapping belongs the ith context
# context_goals = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5]
# context_maps =  [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]

# context_goals = [0]*8 + [1]*9 + [2]*15
# context_maps =  [0]*5 + [1,2,3]*9

# context_goals = [0]*7 + [1]*9 + [2]*69
# context_maps =  [0]*4 + [1,2,3]*27

# test_context_goals = [0,0,1,2]
# test_context_maps =  [0,4,5,6]

# favour joint
# context_goals = [0]*6 + [3]*10
# context_maps =  [0]*4 + [1,2]*6

# test_context_goals = [0,0,3]
# test_context_maps =  [0,4,5]

context_goals = [0]*6 + [3]*28
context_maps =  [0]*4 + [1,2]*15

# intermediate reversal
context_goals = [0]*6 + [3]*16
context_maps =  [0]*4 + [1,2]*15

# minimal full reversal
context_goals = [0]*6 + [3]*20
context_maps =  [0]*4 + [1,2]*15


test_context_goals = [0,0,3]
test_context_maps =  [0,4,5]


start_locations = [(x, y) for x in range(1, 5) for y in range(1, 5)]
context_balance = [4] * (len(context_goals) + len(test_context_goals))

# the hazard rate determines a degree of auto amcorrelation in the context orders. This is
# useful in human studies. The hazard rates is the defined by the probability of a 
# context change after i repeats is f(i)
hazard_rates = [0.5, 0.67, 0.67, 0.75, 1.0, 1.0]

# randomly start the agent somewhere in the middle of the map
task_kwargs = dict(context_balance=context_balance, 
                   context_goals=[goal_locations[g] for g in context_goals], 
                   context_maps=[mapping_definitions[m] for m in context_maps],
                   hazard_rates=hazard_rates, start_locations=start_locations,
                   test_context_goals=[goal_locations[g] for g in test_context_goals], 
                   test_context_maps=[mapping_definitions[m] for m in test_context_maps],
                   grid_world_size=grid_world_size, list_goal_locations=goal_locations.values()
                  )

n_sim = 2 ## run 150 in the paper
seed = 65756
agent_kwargs = dict(discount_rate=0.75, inverse_temperature=5.0)
meta_kwargs = dict(m_biases=[0.0, 0.0])  # log prior for Ind, Joint models, respectively
metarl_kwargs = dict(m_biases=[0.0, 0.0], mixing_lrate=0.2, mixing_temp=5.0)

sim2 = simulate_task(n_sim, task_kwargs, agent_kwargs=agent_kwargs, meta_kwargs=meta_kwargs,
                     metarl_kwargs=metarl_kwargs, alpha=1.0, seed=seed)

# sim2.to_pickle("./AmbigEnv3Results.pkl")


# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # plot_results(sim2, figsize=(9, 4.5))


# with sns.axes_style('ticks'):
#     sns.factorplot(x='Times Seen Context', y='n actions taken', data=sim2[sim2['In goal']],
#           units='Simulation Number', hue='Model', estimator=np.mean,
#           palette='Set2')
#     sns.despine()
    
    
# df0 = sim2[sim2['In goal']].groupby(['Model', 'Simulation Number']).sum()
# cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim2.Model)]
# model = []
# for m in set(sim2.Model):
#     model += [m] * (sim2[sim2.Model == m]['Simulation Number'].max() + 1)
# df1 = pd.DataFrame({
#         'Cumulative Steps Taken': np.concatenate(cum_steps),
#         'Model': model
#     })

# # sns.set_context('talk')
# with sns.axes_style('ticks'):
#     fig, ax = plt.subplots(1, 1, figsize=(2, 3))
#     cc = sns.color_palette('Set2')
#     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette=cc[1:],
#                     order=["Independent", "Joint", 'Meta']
#                     # order=["Independent", "Joint", 'Hierarchical', 'Meta']
#                     )
#     ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
#     ax.plot([-0.5, 3], [ybar, ybar], 'r--')
#     ax.set_ylabel('Total Steps')
#     ax.set_xticklabels(['Indep.', 'Joint', 'Meta'])
#     # ax.set_xticklabels(['Indep.', 'Joint', 'Hier.', 'Meta'])

#     sns.despine()
#     # plt.savefig('sim2_mixed.png', dpi=300, bbox_inches='tight')
    
    
    
# df0 = sim2[sim2['In goal']].groupby(['Model', 'Simulation Number', 'Trial Number']).mean()
# df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
# df0 = df0.rename(index=str, columns={'n actions taken': "Cumulative Steps Taken"})
# df0 = df0[df0['Trial Number'] == df0['Trial Number'].max()]
# # sim1.groupby(['Model', 'Simulation Number']).mean()
# print df0.groupby('Model').mean()['Cumulative Steps Taken']
# print df0.groupby('Model')['Cumulative Steps Taken'].std()
