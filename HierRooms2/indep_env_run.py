import pandas as pd

from model.comp_rooms import RoomsProblem, compute_task_mutual_info
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent#, JointClusteringAgent
from model.generate_env import generate_room_args_indep2 as generate_room_args

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import numpy as np

seed = 500
np.random.seed(seed)


actions = (u'left', u'up', u'down', u'right')
n_a = 8

goal_ids = ["A","B","C","D"]
goal_coods = [(1, 2), (2, 5), (3, 1), (4, 3)]

n_sims = 50

alpha = 1.0
inv_temp = 5.0

mappings_idx = [0]*8 + [1]*5 + [2]*2 + [3]
rewards_idx  = [0]*8 + [1]*5 + [2]*2 + [3]

room_args_list = [None]*n_sims
task_mutual_info = [None]*n_sims

# room_args_list = pd.read_pickle("./analyses/RoomArgs.pkl")


for ii in range(n_sims):
    # # specify mappings, door sequences, and rewards for each room and its sublevels
    room_mappings_idx    = np.array(np.random.permutation(mappings_idx))
    door_sequences_idx   = np.array(np.random.permutation(rewards_idx))
    sublvl_rewards_idx   = np.array(np.random.permutation(rewards_idx))
    sublvl1_mappings_idx = np.array(np.random.permutation(mappings_idx))
    sublvl2_mappings_idx = np.array(np.random.permutation(mappings_idx))
    sublvl3_mappings_idx = np.array(np.random.permutation(mappings_idx))

    room_mappings_idx = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 2, 3, 0, 2, 0, 0])
    door_sequences_idx = np.array([1, 1, 0, 1, 1, 2, 2, 3, 0, 0, 0, 0, 1, 0, 0, 0])

    task_args = [actions, n_a, goal_ids, goal_coods, room_mappings_idx, door_sequences_idx, sublvl_rewards_idx, 
              sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx]

    room_args_list[ii] = generate_room_args(*task_args)


def sim_task(room_args_list, desc='Running Task'):

    results = []
    clusterings_hc = []

    print 'Hierarchical'
    rooms_kwargs = dict()
    generate_kwargs = {
        'evaluate': False,
    }

    for ii in tqdm(range(n_sims), desc=desc):
        room_args = room_args_list[ii]
        task = RoomsProblem(*room_args, **rooms_kwargs)
        agent = HierarchicalAgent(task, inv_temp=inv_temp)
        results_hc, _clusterings_hc = agent.navigate_rooms(**generate_kwargs)
        results_hc[u'Model'] = 'Hierarchical'
        results_hc['Iteration'] = [ii] * len(results_hc)
        results.append(results_hc)
        clusterings_hc.append(_clusterings_hc)


    print 'Independent'
    rooms_kwargs = dict()
    generate_kwargs = {
        'evaluate': False,
    }

    for ii in tqdm(range(n_sims), desc=desc):
        room_args = room_args_list[ii]
        task = RoomsProblem(*room_args, **rooms_kwargs)
        agent = IndependentClusterAgent(task, alpha=alpha, inv_temp=inv_temp)
        results_ic, _ = agent.navigate_rooms(**generate_kwargs)
        results_ic[u'Model'] = 'Independent'
        results_ic['Iteration'] = [ii] * len(results_ic)
        results.append(results_ic)


    print 'Flat'
    rooms_kwargs = dict()
    generate_kwargs = {
        'evaluate': False,
    }

    for ii in tqdm(range(n_sims), desc=desc):
        room_args = room_args_list[ii]
        task = RoomsProblem(*room_args, **rooms_kwargs)
        agent = FlatAgent(task, inv_temp=inv_temp)
        results_fl, _ = agent.navigate_rooms(**generate_kwargs)
        results_fl[u'Model'] = 'Flat'
        results_fl['Iteration'] = [ii] * len(results_fl)
        results.append(results_fl)


    return pd.concat(results)


results = sim_task(room_args_list)

