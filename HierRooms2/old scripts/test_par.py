from mpi4py import MPI

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import random

from model.comp_rooms import RoomsProblem
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent#, JointClusteringAgent

import numpy as np

n_sims = 50
n_rooms = 2

grid_world_size = (6, 6)
subgrid_world_size = (6, 6)

# make it easy, have start locations be the same for each room
start_location = {r: (0,0) for r in range(n_rooms)}

# shall also have start locations be same in each sublvl

door_ids = ["A","B","C","D"]
door_coords = [(1, 2), (2, 5), (3, 1), (4, 3)]
doors = np.array(zip(door_ids, door_coords))
n_doors = len(door_coords)

alpha = 1.0
inv_temp = 5.0


def sim_task(rooms_args, desc='Running Task'):

    results = []
    for ii in tqdm(range(n_sims), desc=desc):
        
        print 'Hierarchical'

        rooms_kwargs = dict()
        
        generate_kwargs = {
             'evaluate': False,
             }

        task = RoomsProblem(*rooms_args, **rooms_kwargs)
        agent = HierarchicalAgent(task, inv_temp=inv_temp)
        results_hc = agent.navigate_rooms(**generate_kwargs)
        results_hc[u'Model'] = 'Hierarchical'
        results_hc['Iteration'] = [ii] * len(results_hc)


        print 'Independent'

        rooms_kwargs = dict()
        
        generate_kwargs = {
#            'prunning_threshold': 1.5,
            'evaluate': False,
            }

        task = RoomsProblem(*rooms_args, **rooms_kwargs)
        agent = IndependentClusterAgent(task, alpha=alpha, inv_temp=inv_temp)
        results_ic = agent.navigate_rooms(**generate_kwargs)
        results_ic[u'Model'] = 'Independent'
        results_ic['Iteration'] = [ii] * len(results_ic)

#        # task = RoomsProblem(*rooms_args, **rooms_kwargs)
#        # agent = JointClusteringAgent(task, alpha=alpha, inv_temp=inv_temp)
#        # results_jc = agent.navigate_rooms(**generate_kwargs)
#        # results_jc[u'Model'] = 'Joint'
#        # results_jc['Iteration'] = [ii] * len(results_jc)
#
        print 'Flat'
        
        rooms_kwargs = dict()

        generate_kwargs = {
#            'prunning_threshold': 10.0,
            'evaluate': False,
            }

        task = RoomsProblem(*rooms_args, **rooms_kwargs)
        agent = FlatAgent(task, inv_temp=inv_temp)
        results_fl = agent.navigate_rooms(**generate_kwargs)
        results_fl[u'Model'] = 'Flat'
        results_fl['Iteration'] = [ii] * len(results_fl)

        results.append(results_hc)
        results.append(results_ic)
        # results.append(results_jc)
        results.append(results_fl)
    return pd.concat(results)


def simulate_one(agent_class, simulation_number, rooms_args, seed=None, desc='Running Task'):
    if seed is not None:
        np.random.seed(seed)

    rooms_kwargs = dict()
        
    generate_kwargs = {'evaluate': False}

    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = agent_class(task, inv_temp=inv_temp)
    results = agent.navigate_rooms(**generate_kwargs)
    results['Iteration'] = [simulation_number] * len(results_hc)
    
    return results


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()


random.seed(datetime.now())
seed0 = random.randint(0,10000)*(rank+1)
seed = int(seed0)
np.random.seed(seed)


# generate list of tasks for each sim and scatter data
# master process
if rank == 0:
    rooms_args_list = []
    for kk in range(n_sims):
        execfile("generate_envs.py")

        door_locations = {r: doors[door_order[r]] for r in range(n_rooms)}
        sublvl_door_locations = {r: {door : door_coords[ii] for ii, door in enumerate(door_ids)} for r in range(n_rooms)}
        rooms_args = [room_mappings, start_location, door_locations, sublvl_mappings, subreward_function, sublvl_door_locations]

        rooms_args_list.append((kk, rooms_args))

    q, r = divmod(n_sims, n_procs)
    counts = [q + 1 if p < r else q for p in range(n_procs)]    

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(n_procs)]
    ends = [sum(counts[:p+1]) for p in range(n_procs)]

    rooms_args_list = [rooms_args_list[starts[p]:ends[p]] for p in range(n_procs)]

# worker process
else:
    rooms_args_list = None      

rooms_args_list = comm.scatter(rooms_args_list, root=0)



# Run sims

rooms_kwargs = dict()
generate_kwargs = {
    'evaluate': False,
}


# Hierarchical Agent
results_hc = []
for kk, rooms_args in rooms_args_list:
    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = HierarchicalAgent(task, inv_temp=inv_temp)
    _results = agent.navigate_rooms(**generate_kwargs)
    _results[u'Model'] = 'Hierarchical'
    _results['Iteration'] = [kk] * len(_results)
    results_hc.append(_results)

# Independent Agent
results_ic = []
for kk, rooms_args in rooms_args_list:
    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = IndependentClusterAgent(task, inv_temp=inv_temp)
    _results = agent.navigate_rooms(**generate_kwargs)
    _results[u'Model'] = 'Independent'
    _results['Iteration'] = [kk] * len(_results)
    results_ic.append(_results)

# Flat Agent
results_fl = []
for kk, rooms_args in rooms_args_list:
    task = RoomsProblem(*rooms_args, **rooms_kwargs)
    agent = FlatAgent(task, inv_temp=inv_temp)
    _results = agent.navigate_rooms(**generate_kwargs)
    _results[u'Model'] = 'Flat'
    _results['Iteration'] = [kk] * len(_results)
    results_fl.append(_results)


gathered_results = comm.gather((results_hc, results_ic, results_fl), root=0)


if rank == 0:
    results_hc, results_ic, results_fl = [], [], []
    
    for results in gathered_results:
        _results_hc, _results_ic, _results_fl = results

        results_hc += _results_hc
        results_ic += _results_ic
        results_fl += _results_fl
    
    results = pd.concat(results_hc + results_ic + results_fl)
    
    results.to_pickle("./HierarchicalRooms.pkl")