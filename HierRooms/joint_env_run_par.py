from mpi4py import MPI

import pandas as pd
import numpy as np
import pickle

from model.comp_rooms import RoomsProblem, compute_task_info_measures
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent
from model.generate_env import generate_room_args_common_mapping_sets as generate_room_args


n_sims = 50

actions = (u'left', u'up', u'down', u'right')
n_a = 8

goal_ids = ["A","B","C","D"]
goal_coods = [(1, 2), (2, 5), (3, 1), (4, 3)]


# specify mappings, door sequences, and rewards for each room and its sublevels
room_mappings_idx =    np.array(range(4)*4)
door_sequences_idx =   np.array(range(4)*4)
sublvl_rewards_idx =   np.array(range(4)*4)
sublvl1_mappings_idx = np.array(range(4)*4)
sublvl2_mappings_idx = np.array(range(4)*4)
sublvl3_mappings_idx = np.array(range(4)*4)

task_args = [actions, n_a, goal_ids, goal_coods, room_mappings_idx, door_sequences_idx, sublvl_rewards_idx, 
             sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx]

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
    room_args_list = []
    for kk in range(n_sims):
        room_args = generate_room_args(*task_args)
        room_args_list.append((kk, room_args))

    task_info_measures = [ compute_task_info_measures(*room_args[1]) for room_args in room_args_list]

    q, r = divmod(n_sims, n_procs)
    counts = [q + 1 if p < r else q for p in range(n_procs)]    

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(n_procs)]
    ends = [sum(counts[:p+1]) for p in range(n_procs)]

    room_args_list = [room_args_list[starts[p]:ends[p]] for p in range(n_procs)]

# worker process
else:
    room_args_list = None      

room_args_list = comm.scatter(room_args_list, root=0)



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
for sim_number, room_args in room_args_list:
    task = RoomsProblem(*room_args, **room_kwargs)
    agent = FlatAgent(task, inv_temp=inv_temp)
    _results, _ = agent.navigate_rooms()
    _results[u'Model'] = 'Flat'
    _results['Iteration'] = [sim_number] * len(_results)
    results_fl.append(_results)

_results_fl = comm.gather(results_fl, root=0)

# get mean steps for flat agent
if rank == 0:
    results_fl = []
    for result in _results_fl:
        results_fl += result

    _results_fl = pd.concat(results_fl)
    success_trials = _results_fl[_results_fl['Success'] == True]
    mean_steps_fl = success_trials['Cumulative Steps'].mean()    
else:   
    mean_steps_fl = None
 
mean_steps_fl = comm.bcast(mean_steps_fl, root=0)
assert mean_steps_fl is not None

# Hierarchical Agent
results_hc = []
clusterings_hc = []
for sim_number, room_args in room_args_list:
    task = RoomsProblem(*room_args, **room_kwargs)
    agent = HierarchicalAgent(task, inv_temp=inv_temp, max_steps=mean_steps_fl)
    _results, _clusterings_hc = agent.navigate_rooms()
    _results[u'Model'] = 'Hierarchical'
    _results['Iteration'] = [sim_number] * len(_results)
    results_hc.append(_results)
    clusterings_hc.append(_clusterings_hc)

# Independent Agent
results_ic = []
for sim_number, room_args in room_args_list:
    task = RoomsProblem(*room_args, **room_kwargs)
    agent = IndependentClusterAgent(task, inv_temp=inv_temp, max_steps=mean_steps_fl)
    _results, _ = agent.navigate_rooms()
    _results[u'Model'] = 'Independent'
    _results['Iteration'] = [sim_number] * len(_results)
    results_ic.append(_results)

_results_hc = comm.gather(results_hc, root=0)
_results_ic = comm.gather(results_ic, root=0)
_clusterings_hc = comm.gather(clusterings_hc, root=0)

if rank == 0:
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
    
    results.to_pickle("./HierarchicalRooms_joint.pkl")
    pickle.dump( clusterings_hc, open( "HierarchicalRoomsClusterings_joint_hc.pkl", "wb" ) )
    pickle.dump( compute_task_info_measures, open ("TaskInfoMeasures_joint_hc.pkl", "wb"))
