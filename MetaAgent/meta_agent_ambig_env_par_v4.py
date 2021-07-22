#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:19:14 2020

@author: rex
"""


from model import par_simulate_mixed_task as simulate_task

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
    4:(0, 3),
    5:(5, 3),
}

# define the mappings for each context, where the ith mapping belongs the ith context
# context_goals = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5]
# context_maps =  [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]

context_goals = [0]*8 + [1]*9 + [2]*15
context_maps =  [0]*5 + [1,2,3]*9

test_context_goals = [0,0,1,2]
test_context_maps =  [0,4,5,6]

start_locations = [(x, y) for x in range(1, 5) for y in range(1, 5)]
print len(context_goals) + len(test_context_goals)
context_balance = [4] * (len(context_goals) + len(test_context_goals))

# the hazard rate determines a degree of auto correlation in the context orders. This is
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
                   grid_world_size=grid_world_size
                  )

n_sim = 150 ## run 150 in the paper
seed = 65756
agent_kwargs = dict(discount_rate=0.75, inverse_temperature=5.0)
meta_kwargs = dict(m_biases=[0.0, 0.0])  # log prior for Ind, Joint models, respectively
metarl_kwargs = dict(m_biases=[0.0, 0.0], mixing_lrate=0.2, mixing_temp=5.0)

sim2 = simulate_task(n_sim, task_kwargs, agent_kwargs=agent_kwargs, meta_kwargs=meta_kwargs,
                     metarl_kwargs=metarl_kwargs, alpha=1.0, seed=seed)

sim2.to_pickle("./AmbigEnv3Results.pkl")


