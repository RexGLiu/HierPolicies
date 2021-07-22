#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:19:14 2020

@author: rex
"""

from mpi4py import MPI

from model.rooms_problem import RoomsProblem
from model.rooms_agents import IndependentClusterAgent, JointClusteringAgent, FlatAgent, HierarchicalAgent

import numpy as np
import pandas as pd

grid_world_size = (6, 6)

# make it easy, have the door and start locations be the same for each room
start_location = {r: (0,0) for r in range(9)}

# make it easy, each door is in the same spot
door_locations = {r: {'A':(5, 5), 'B':(5, 0), 'C':(0, 5)} for r in range(9)}

# this is the context: goal function thing, 
sucessor_function = {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 2, "B": 0, "C": 0},
    2: {"A": 3, "B": 0, "C": 0},
    3: {"A": 4, "B": 0, "C": 0},
    4: {"A": 5, "B": 0, "C": 0},
    5: {"A": None, "B": 0, "C": 0}, # signifies the end!
}

reward_function = {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 1, "B": 0, "C": 0},
    2: {"A": 1, "B": 0, "C": 0},
    3: {"A": 1, "B": 0, "C": 0},
    4: {"A": 1, "B": 0, "C": 0},
    5: {"A": 1, "B": 0, "C": 0},
}

mapping_listings = [
    {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    {1: u'left', 2: u'up', 3: u'down', 0: u'right'},
    {2: u'left', 3: u'up', 0: u'down', 1: u'right'},
    {3: u'left', 0: u'up', 1: u'down', 2: u'right'},
    {4: u'left', 5: u'up', 6: u'down', 7: u'right'},
    {5: u'left', 6: u'up', 7: u'down', 4: u'right'},
    {6: u'left', 7: u'up', 4: u'down', 5: u'right'},
    {7: u'left', 4: u'up', 5: u'down', 6: u'right'}
]

# mapping_listings = [
#     {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
#     {1: u'left', 2: u'up', 3: u'down', 4: u'right'},
#     {2: u'left', 3: u'up', 4: u'down', 5: u'right'},
#     {3: u'left', 4: u'up', 5: u'down', 6: u'right'},
#     {4: u'left', 5: u'up', 6: u'down', 7: u'right'},
#     {5: u'left', 6: u'up', 7: u'down', 0: u'right'},
#     {6: u'left', 7: u'up', 0: u'down', 1: u'right'},
#     {7: u'left', 0: u'up', 1: u'down', 2: u'right'}
# ]

n_mappings = len(mapping_listings)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

n_sims = 150 ## run 150 in the paper


generate_kwargs = {
    'prunning_threshold': 10.0,
    'evaluate': False,
}

rooms_kwargs = dict()

alpha = 1.0
inv_temp = 5.0

if rank == 0:
    seed = 65756
    np.random.seed(seed)
    rand_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_procs)
else:
    rand_seeds = None
    
seed = comm.scatter(rand_seeds, root=0)
np.random.seed(seed)

q, r = divmod(n_sims, n_procs)
if rank < r:
    n_local_sims = q + 1
    sim_offset = n_local_sims*rank
else:
    n_local_sims = q
    sim_offset = (q+1)*r + q*(rank-r)


results_jc = []
results_ic = []
results_fl = []
results_h = []

for kk in range(n_local_sims):
    sim_number = sim_offset+kk
    
    mappings = {ii: mapping_listings[ii] for ii in np.random.permutation(n_mappings)}
    room_mappings = {
            0: mappings[0],
            1: mappings[0],
            2: mappings[1],
            3: mappings[1],
            4: mappings[2],
            5: mappings[2],
    }

    rooms_args = list([room_mappings, sucessor_function, reward_function, start_location,
                  door_locations])
    
    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = HierarchicalAgent(task, alpha_r=alpha, alpha_hi=alpha, alpha_lo=alpha, inv_temp=inv_temp)
    _results = agent.navigate_rooms(**generate_kwargs)
    _results['Iteration'] = [sim_number] * len(_results)
    results_h.append(_results)

    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = IndependentClusterAgent(task, alpha=alpha, inv_temp=inv_temp)
    _results = agent.navigate_rooms(**generate_kwargs)
    _results['Iteration'] = [sim_number] * len(_results)
    results_ic.append(_results)

    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = JointClusteringAgent(task, alpha=alpha, inv_temp=inv_temp)
    _results = agent.navigate_rooms(**generate_kwargs)
    _results['Iteration'] = [sim_number] * len(_results)
    results_jc.append(_results)

    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = FlatAgent(task, inv_temp=inv_temp)
    _results = agent.navigate_rooms(**generate_kwargs)
    _results['Iteration'] = [sim_number] * len(_results)
    results_fl.append(_results)



_results_h = comm.gather(results_h, root=0)
if rank == 0:
    results_h = []
    for result in _results_h:
        results_h += result
    results_h = pd.concat(results_h)
    results_h[u'Model'] = 'Hierarchical'
    del _results_h

_results_jc = comm.gather(results_jc, root=0)
if rank == 0:
    results_jc = []
    for result in _results_jc:
        results_jc += result
    results_jc = pd.concat(results_jc)
    results_jc[u'Model'] = 'Joint'
    del _results_jc

_results_ic = comm.gather(results_ic, root=0)
if rank == 0:
    results_ic = []
    for result in _results_ic:
        results_ic += result
    results_ic = pd.concat(results_ic)
    results_ic[u'Model'] = 'Independent'
    del _results_ic

_results_fl = comm.gather(results_fl, root=0)
if rank == 0:
    results_fl = []
    for result in _results_fl:
        results_fl += result
    results_fl = pd.concat(results_fl)
    results_fl[u'Model'] = 'Flat'
    del _results_fl

    results = pd.concat([results_jc, results_ic, results_fl, results_h])
    results.to_pickle("./DiabolicalRoomsResults.pkl")
    