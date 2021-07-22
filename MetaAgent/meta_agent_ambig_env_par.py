#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:50:29 2020

@author: rex
"""

#from model.gridworld import make_task
#from model import list_entropy, mutual_information, plot_results
from model import plot_results
from model import par_simulate_mixed_task as simulate_task

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
    # 2:(5, 0),
    # 3:(5, 5),
}

# assign goals and mappings to contexts

#minimize the alpha for each goal sequence
from scipy.optimize import minimize


N = 10
n_conflict = range(N/2, N)
compositionality_benefit = list()
compositionality_benefit_fa = list()
mutual_information_list = list()

n_sim = 150 ## run 150 in the paper
seed = 234234

# alternative goals located on edge of map
alternative_goals = [ (x,0) for x in range (1,6)] + [ (x,5) for x in range (1,6)] + [ (0,y) for y in range (1,5)] + [ (5,y) for y in range (1,5)]
np.random.shuffle(alternative_goals)

for i in range(N):
    goal_locations[i+2] = alternative_goals[i]

for n in n_conflict:
    print n
    
    context_goals = [0] * N + [1] * N 
    context_maps = [0] * (n) + [1] * (N-n) + [1] * n + [0] * (N-n)
    
    goal_idx = 2
    for ii in range(len(context_maps)):
        if context_maps[ii] != context_goals[ii]:
            context_goals[ii] = goal_idx
            goal_idx += 1            
    
    print 'context goals: ', context_goals
    print 'context goals: ', context_maps
    
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
                  grid_world_size=grid_world_size,
                  )

    agent_kwargs = dict(discount_rate=0.75, inverse_temperature=5.0)
    meta_kwargs = dict(m_biases=[0.0, 0.0])  # log prior for Ind, Joint models, respectively
    metarl_kwargs = dict(m_biases=[0.0, 0.0], mixing_lrate=0.2, mixing_temp=5.0)

    sim1 = simulate_task(n_sim, task_kwargs, agent_kwargs=agent_kwargs, alpha=1.0, seed=seed,
                    meta_kwargs=meta_kwargs, metarl_kwargs=metarl_kwargs)

    sim1.to_pickle("./AmbigEnvResults_"+str(n)+".pkl")