#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:50:29 2020

@author: Rex Liu (based heavily(!) on Nick Franklin's code)
"""

from model import plot_results
from model import simulate_mixed_task as simulate_task

# define all of the task parameters
grid_world_size = (6, 6)

# define mapping between primitive action (a in [0, 1]) and cardinal movements
# (left, right, up down)
mapping_definitions = {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
}

# define goal locations in (x, y) coordinate space
goal_locations = {
    0:(0, 0),
    1:(0, 5),
    2:(5, 0),
    3:(5, 5),
}

# assign goals and mappings to contexts
context_maps  = [0,1,2,3]*2   + [4,5,6,7]*2
context_goals = [0]*4 + [1]*4 + [2]*4 + [3]*4

# randomly start the agent somewhere in the middle of the map
start_locations = [(x, y) for x in range(1, 5) for y in range(1, 5)]

# the number of times each context is shown
context_balance = [4] * len(context_goals)

# the hazard rate determines a degree of auto correlation in the context orders. This is
# useful in human studies. The hazard rates is the defined by the probability of a 
# context change after i repeats is f(i)
hazard_rates = [0.5, 0.67, 0.67, 0.75, 1.0, 1.0]

task_kwargs = dict(context_balance=context_balance, 
                   context_goals=[goal_locations[g] for g in context_goals], 
                   context_maps=[mapping_definitions[m] for m in context_maps],
                   hazard_rates=hazard_rates, start_locations=start_locations,
                   grid_world_size=grid_world_size, list_goal_locations=goal_locations.values(),
                   )


n_sim = 2 ## run 150 in the paper
seed = 234234
agent_kwargs = dict(discount_rate=0.75, inverse_temperature=5.0)
meta_kwargs = dict(m_biases=[0.0, 0.0])  # log prior for Ind, Joint models, respectively
metarl_kwargs = dict(m_biases=[0.0, 0.0], mixing_lrate=0.2, mixing_temp=5.0)

sim1 = simulate_task(n_sim, task_kwargs, agent_kwargs=agent_kwargs, alpha=1.0, seed=seed,
                     meta_kwargs=meta_kwargs, metarl_kwargs=metarl_kwargs)

sim1.to_pickle("./CondIndepEnvResults.pkl")
