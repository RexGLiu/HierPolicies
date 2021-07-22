import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from model.comp_rooms import RoomsProblem
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent#, JointClusteringAgent
from model.generate_env import generate_room_args

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import numpy as np

seed = 500
np.random.seed(seed)


actions = (u'left', u'up', u'down', u'right')
n_a = 8

goal_ids = ["A","B","C","D"]
goal_coods = [(1, 2), (2, 5), (3, 1), (4, 3)]


# specify mappings, door sequences, and rewards for each room and its sublevels
room_mappings_idx =    np.array([0,0,0,1,1,1,1,2,3,4,5,6,2,3,4,5])
door_sequences_idx =   np.array([0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,3])
sublvl_rewards_idx =   np.array([0,0,0,1,1,1,1,2,2,3,3,3,2,2,3,3])
sublvl1_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*3)))
sublvl2_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*3)))
sublvl3_mappings_idx = np.array([0,0,0,1,2,3,4] + list(np.random.permutation([5,6,7]*3)))



# specify mappings, door sequences, and rewards for each room and its sublevels
room_mappings_idx =    np.array([0,1])
door_sequences_idx =   np.array([0,1])
sublvl_rewards_idx =   np.array([0,1])
sublvl1_mappings_idx = np.array([0,1])
sublvl2_mappings_idx = np.array([0,1])
sublvl3_mappings_idx = np.array([0,1])


task_args = [actions, n_a, goal_ids, goal_coods, room_mappings_idx, door_sequences_idx, sublvl_rewards_idx, 
             sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx]

room_args = generate_room_args(*task_args)

n_sims = 2

alpha = 1.0
inv_temp = 5.0



def sim_task(room_args, desc='Running Task'):

    results = []
    clusterings_hc = []

    print 'Hierarchical'
    rooms_kwargs = dict()
    generate_kwargs = {
        'evaluate': False,
    }

    for ii in tqdm(range(n_sims), desc=desc):
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
        task = RoomsProblem(*room_args, **rooms_kwargs)
        agent = FlatAgent(task, inv_temp=inv_temp)
        results_fl, _ = agent.navigate_rooms(**generate_kwargs)
        results_fl[u'Model'] = 'Flat'
        results_fl['Iteration'] = [ii] * len(results_fl)
        results.append(results_fl)


    return pd.concat(results)


results = sim_task(room_args)

