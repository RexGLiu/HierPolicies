#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:50:29 2020

@author: rex
"""

from mpi4py import MPI

from model import make_task, JointClustering, IndependentClusterAgent, FlatControlAgent, MetaAgent, HierarchicalAgent
from model import simulate_one

import numpy as np
import pandas as pd
import pickle

# define all of the task parameters
grid_world_size = (6, 6)

# define mapping between primitive action (a in [0, 1]) and cardinal movements
# (left, right, up down)
mapping_definitions = {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {1: u'left', 2: u'up', 3: u'down', 0: u'right'},
    2: {2: u'left', 3: u'up', 0: u'down', 1: u'right'},
    3: {3: u'left', 0: u'up', 1: u'down', 2: u'right'},
    4: {4: u'left', 5: u'up', 6: u'down', 7: u'right'},
    5: {5: u'left', 6: u'up', 7: u'down', 4: u'right'},
    6: {6: u'left', 7: u'up', 4: u'down', 5: u'right'},
    7: {7: u'left', 4: u'up', 5: u'down', 6: u'right'}
}


# define goal locations in (x, y) coordinate space
goal_locations = {
    0:(0, 0),
    1:(0, 5),
    2:(5, 0),
    3:(5, 5),
}


# assign goals and mappings to contexts
context_maps  = [0]*3 + [1]*3 + [2,3,4,5,6]*2
context_goals = [0]*3 + [3]*3 + [1]*5 + [2]*5

# randomly start the agent somewhere in the middle of the map
start_locations = [(x, y) for x in range(1, 5) for y in range(1, 5)]

# the number of times each context is shown
context_balance = [4] * len(context_goals)

# the hazard rate determines a degree of auto correlation in the context orders. This is
# useful in human studies. The hazard rates is the defined by the probability of a 
# context change after i repeats is f(i)
hazard_rates = [0.5, 0.67, 0.67, 0.75, 1.0, 1.0]

# collate task params into a dict
task_kwargs = dict(context_balance=context_balance, 
                   context_goals=[goal_locations[g] for g in context_goals], 
                   context_maps=[mapping_definitions[m] for m in context_maps],
                   hazard_rates=hazard_rates, start_locations=start_locations,
                   grid_world_size=grid_world_size, list_goal_locations=goal_locations.values(),
                   )

n_sims = 150 ## run 150 in the paper

# collate agent params into a dict
agent_kwargs = dict(discount_rate=0.75, inverse_temperature=5.0, alpha=1.0)
meta_kwargs = dict(agent_kwargs)
meta_kwargs['m_biases'] = [0.0, 0.0]  # log prior for Ind, Joint models, respectively
hier_agent_kwargs = dict(agent_kwargs)
del hier_agent_kwargs['alpha']

# set-up parallel processing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

# randomly select random seeds for each processor in processor 0 and then scatter to all other processors
if rank == 0:
    seed = 234234
    np.random.seed(seed)
    rand_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_procs)
else:
    rand_seeds = None
    
seed = comm.scatter(rand_seeds, root=0)
np.random.seed(seed)

# compute number of trials to run on each processor
q, r = divmod(n_sims, n_procs)
if rank < r:
    n_local_sims = q + 1
    sim_offset = n_local_sims*rank   # used for determining trial number
else:
    n_local_sims = q
    sim_offset = (q+1)*r + q*(rank-r)   # used for determining trial number

# lists for storing local sim results
results_jc = [None] * n_local_sims
results_ic = [None] * n_local_sims
results_fl = [None] * n_local_sims
results_mx = [None] * n_local_sims
results_h = [None] * n_local_sims

clusterings_h = [None] * n_local_sims

# iterate through trials
for kk in range(n_local_sims):
    sim_number = sim_offset+kk
    
    task = make_task(**task_kwargs)  # create task for current trial
    results_h[kk], clusterings_h[kk] = simulate_one(HierarchicalAgent, sim_number, task, agent_kwargs=hier_agent_kwargs)

    task.reset()  # reset task and reuse for all other agents
    results_mx[kk], _ = simulate_one(MetaAgent, sim_number, task, agent_kwargs=meta_kwargs)

    task.reset()
    results_jc[kk], _ = simulate_one(JointClustering, sim_number, task, agent_kwargs=agent_kwargs)

    task.reset()
    results_ic[kk], _ = simulate_one(IndependentClusterAgent, sim_number, task, agent_kwargs=agent_kwargs)

    task.reset()
    results_fl[kk], _ = simulate_one(FlatControlAgent, sim_number, task, agent_kwargs=agent_kwargs)

# gather all results to processor 0 for collation and saving
# hierarchical agent results
_results_h = comm.gather(results_h, root=0)
if rank == 0:
    results_h = []
    for result in _results_h:
        results_h += result
    results_h = pd.concat(results_h)
    results_h['Model'] = ['Hierarchical'] * len(results_h)
    del _results_h

_clusterings_h = comm.gather(clusterings_h, root=0)
if rank == 0:
    clusterings_h = []
    for clusterings in _clusterings_h:
        clusterings_h += clusterings
    
    pickle.dump( clusterings_h, open( "AmbigEnvClusterings_h_mixed.pkl", "wb" ) )
    del _clusterings_h, clusterings_h

# mixed agent results
_results_mx = comm.gather(results_mx, root=0)
if rank == 0:
    results_mx = []
    for result in _results_mx:
        results_mx += result
    results_mx = pd.concat(results_mx)
    results_mx['Model'] = ['Meta'] * len(results_mx)
    del _results_mx

# joint agent results
_results_jc = comm.gather(results_jc, root=0)
if rank == 0:
    results_jc = []
    for result in _results_jc:
        results_jc += result
    results_jc = pd.concat(results_jc)
    results_jc['Model'] = ['Joint'] * len(results_jc)
    del _results_jc

# independent agent results
_results_ic = comm.gather(results_ic, root=0)
if rank == 0:
    results_ic = []
    for result in _results_ic:
        results_ic += result
    results_ic = pd.concat(results_ic)
    results_ic['Model'] = ['Independent'] * len(results_ic)
    del _results_ic

# flat agent results
_results_fl = comm.gather(results_fl, root=0)
if rank == 0:
    results_fl = []
    for result in _results_fl:
        results_fl += result
    results_fl = pd.concat(results_fl)
    results_fl['Model'] = ['Flat'] * len(results_fl)
    del _results_fl

    # save results to file
    results = pd.concat([results_jc, results_ic, results_fl, results_mx, results_h])
    results.to_pickle("./AmbigEnvResults_mixed.pkl")
