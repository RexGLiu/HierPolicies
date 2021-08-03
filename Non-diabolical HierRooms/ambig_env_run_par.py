from mpi4py import MPI

import pandas as pd
import numpy as np
import pickle

from model.comp_rooms import make_task
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent
from model.generate_env import generate_room_args as generate_room_args

n_sims = 50

grid_world_size = (6, 6)

actions = (u'left', u'up', u'down', u'right')
n_a = 8

goal_ids = ["A","B","C","D"]
goal_coords = [(1, 2), (2, 5), (3, 1), (4, 3)]

alpha = 1.0
inv_temp = 5.0


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

if rank == 0:
    seed = 65756
    np.random.seed(seed)
    rand_seeds = np.random.randint(np.iinfo(np.int32).max, size=n_procs)
else:
    rand_seeds = None
    
seed = comm.scatter(rand_seeds, root=0)
np.random.seed(seed)


# generate list of tasks for each sim and scatter data
# master process
if rank == 0:
    task_list = [0]*n_sims
    task_info_measures = [0]*n_sims
    for ii in range(n_sims):
        # specify mappings, door sequences, and rewards for each room and its sublevels
        room_mappings_idx =    np.array([0,0,0,1,1,1,1,2,3,4,5,6,2,3,4,5])
        door_sequences_idx =   np.array([0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,3])
        sublvl_rewards_idx =   np.array([0,0,0,1,1,1,1,2,2,3,3,3,2,2,3,3])
        sublvl1_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*3)))
        sublvl2_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*3)))
        sublvl3_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*3)))


        # room_mappings_idx =    np.array([0,0,0,1,1,1,1,2,3,4,5,6,2,3,4,5,6,2,3,4,5,6])
        # door_sequences_idx =   np.array([0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3])
        # sublvl_rewards_idx =   np.array([0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,3,3,2,3,2,3,2])
        # sublvl1_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*5)))
        # sublvl2_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*5)))
        # sublvl3_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*5)))


        context_balance = [4] * (len(room_mappings_idx))
        hazard_rates = [0.5, 0.67, 0.67, 0.75, 1.0, 1.0]

        task_kwargs = dict(context_balance=context_balance, 
                   n_actions=n_a,
                   actions=actions,
                   goal_ids=goal_ids,
                   goal_coords=goal_coords,
                   room_mappings_idx=room_mappings_idx,
                   door_sequences_idx=door_sequences_idx,
                   sublvl_rewards_idx=sublvl_rewards_idx,
                   sublvl1_mappings_idx=sublvl1_mappings_idx,
                   sublvl2_mappings_idx=sublvl2_mappings_idx,
                   sublvl3_mappings_idx=sublvl3_mappings_idx,
                   hazard_rates=hazard_rates,
                   grid_world_size=grid_world_size,
                   calc_info_measures = True,
                   generate_room_args = generate_room_args
                  )

        task_list[ii], task_info_measures[ii] = make_task(**task_kwargs)

    q, r = divmod(n_sims, n_procs)
    counts = [q + 1 if p < r else q for p in range(n_procs)]    

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(n_procs)]
    ends = [sum(counts[:p+1]) for p in range(n_procs)]

    task_list = [task_list[starts[p]:ends[p]] for p in range(n_procs)]

# worker process
else:
    task_list = None      

task_list = comm.scatter(task_list, root=0)



# Run sims
room_kwargs = dict()

q, r = divmod(n_sims, n_procs)
if rank < r:
    n_local_sims = q + 1
    sim_offset = n_local_sims*rank
else:
    n_local_sims = q
    sim_offset = (q+1)*r + q*(rank-r)


# Run flat agent first
results_fl = []
for kk, task in enumerate(task_list):
    sim_number = sim_offset+kk

    task.reset()
    agent = FlatAgent(task, inv_temp=inv_temp)
    _results, _ = agent.navigate_rooms()
    _results[u'Model'] = 'Flat'
    _results['Simulation Number'] = [sim_number] * len(_results)
    results_fl.append(_results)

# Hierarchical Agent
results_hc = []
clusterings_hc = []
for kk, task in enumerate(task_list):
    sim_number = sim_offset+kk

    task.reset()
    agent = HierarchicalAgent(task, inv_temp=inv_temp)
    _results, _clusterings_hc = agent.navigate_rooms()
    _results[u'Model'] = 'Hierarchical'
    _results['Simulation Number'] = [sim_number] * len(_results)
    results_hc.append(_results)
    clusterings_hc.append(_clusterings_hc)

# Independent Agent
results_ic = []
for kk, task in enumerate(task_list):
    sim_number = sim_offset+kk

    task.reset()
    agent = IndependentClusterAgent(task, inv_temp=inv_temp)
    _results, _ = agent.navigate_rooms()
    _results[u'Model'] = 'Independent'
    _results['Simulation Number'] = [sim_number] * len(_results)
    results_ic.append(_results)

_results_fl = comm.gather(results_fl, root=0)
_results_hc = comm.gather(results_hc, root=0)
_results_ic = comm.gather(results_ic, root=0)
_clusterings_hc = comm.gather(clusterings_hc, root=0)

if rank == 0:
    results_fl = []
    for result in _results_fl:
        results_fl += result

    results_hc = []
    for result in _results_hc:
        results_hc += result

    results_ic = []
    for result in _results_ic:
        results_ic += result
        
    clusterings_hc = []
    for clusterings in _clusterings_hc:
        clusterings_hc += clusterings

    results = pd.concat(results_hc + results_ic + results_fl) 
    
    results.to_pickle("./HierarchicalRooms_ambig.pkl")
    pickle.dump( clusterings_hc, open( "HierarchicalRoomsClusterings_ambig_hc.pkl", "wb" ) )
    pickle.dump( task_info_measures, open ("TaskInfoMeasures_ambig_hc.pkl", "wb"))

