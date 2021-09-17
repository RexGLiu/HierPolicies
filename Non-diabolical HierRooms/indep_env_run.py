import pandas as pd
from tqdm import tqdm
import numpy as np

from model.comp_rooms import make_task
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent
from model.generate_env import generate_room_args_common_mapping_sets as generate_room_args

seed = 500
np.random.seed(seed)


grid_world_size = (6, 6)

actions = (u'left', u'up', u'down', u'right')
n_a = 8

goal_ids = ["A","B","C","D"]
goal_coords = [(1, 2), (2, 5), (3, 1), (4, 3)]

alpha = 1.0
inv_temp = 5.0
min_particles = 100
max_particles = 10000

n_sims = 2

mappings_idx = [0]*8 + [1]*5 + [2]*2 + [3]
rewards_idx  = [0]*8 + [1]*5 + [2]*2 + [3]

mappings_idx = [0] + [1] + [2] + [3]
rewards_idx  = [0] + [1] + [2] + [3]


task_list = [None]*n_sims
task_info_measures = [None]*n_sims
for ii in range(n_sims):
    # specify mappings, door sequences, and rewards for each room and its sublevels
    room_mappings_idx    = np.array(np.random.permutation(mappings_idx))
    door_sequences_idx   = np.array(np.random.permutation(rewards_idx))
    sublvl_rewards_idx   = np.array(np.random.permutation(rewards_idx))
    sublvl1_mappings_idx = np.array(np.random.permutation(mappings_idx))
    sublvl2_mappings_idx = np.array(np.random.permutation(mappings_idx))
    sublvl3_mappings_idx = np.array(np.random.permutation(mappings_idx))

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
                   replacement=True,
                   generate_room_args = generate_room_args
                  )

    task_list[ii], task_info_measures[ii] = make_task(**task_kwargs)


def sim_task(task_list, desc='Running Task'):

    results = []
    clusterings_hc = []

    print 'Hierarchical'
    for ii in tqdm(range(n_sims), desc=desc):
        task = task_list[ii]
        agent = HierarchicalAgent(task, inv_temp=inv_temp, min_particles=min_particles, max_particles=max_particles)
        results_hc, _clusterings_hc = agent.navigate_rooms()
        results_hc[u'Model'] = 'Hierarchical'
        results_hc['Iteration'] = [ii] * len(results_hc)
        results.append(results_hc)
        clusterings_hc.append(_clusterings_hc)


    print 'Independent'
    for ii in tqdm(range(n_sims), desc=desc):
        task = task_list[ii]
        task.reset()
        agent = IndependentClusterAgent(task, alpha=alpha, inv_temp=inv_temp, min_particles=min_particles, max_particles=max_particles)
        results_ic, _ = agent.navigate_rooms()
        results_ic[u'Model'] = 'Independent'
        results_ic['Iteration'] = [ii] * len(results_ic)
        results.append(results_ic)


    print 'Flat'
    for ii in tqdm(range(n_sims), desc=desc):
        task = task_list[ii]
        task.reset()
        agent = FlatAgent(task, inv_temp=inv_temp)
        results_fl, _ = agent.navigate_rooms()
        results_fl[u'Model'] = 'Flat'
        results_fl['Iteration'] = [ii] * len(results_fl)
        results.append(results_fl)

    return pd.concat(results)


results = sim_task(task_list)

