import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
import random

from model.comp_rooms import RoomsProblem
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent#, JointClusteringAgent

from joblib import Parallel, delayed
import multiprocessing

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import numpy as np

execfile("generate_envs.py")



#mappings = {
#    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
#    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
#    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
#    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
#}
#
#room_mappings = {
#    0: mappings[0],
#    1: mappings[0],
#    2: mappings[0],
#    3: mappings[0],
#    4: mappings[0],
#    5: mappings[0],
#}
#
#
#sublvl_mappings = {0: {
#    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
#    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
#    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
#    },
#    1: {
#    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
#    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
#    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
#    },
#    2: {
#    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
#    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
#    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
#    },
##    3: {
##    0: {4: u'up', 8: u'left', 6: u'right', 7: u'down'},
##    1: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
##    2: {1: u'left', 0: u'up', 9: u'right', 2: u'down'},
##    },
##    4: {
##    0: {4: u'up', 8: u'left', 6: u'right', 7: u'down'},
##    1: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
##    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
##    },
##    5: {
##    0: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
##    1: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
##    2: {1: u'left', 0: u'up', 9: u'right', 2: u'down'},
##    }
#    3: {
#    0: {0: u'up', 8: u'left', 3: u'right', 1: u'down'},
#    1: {2: u'left', 9: u'up', 1: u'down', 3: u'right'},
#    2: {1: u'left', 5: u'up', 2: u'right', 8: u'down'},
#    },
#    4: {
#    0: {0: u'up', 8: u'left', 3: u'right', 1: u'down'},
#    1: {2: u'left', 9: u'up', 1: u'down', 3: u'right'},
#    2: {1: u'left', 5: u'up', 2: u'right', 8: u'down'},
#    },
#    5: {
#    0: {0: u'up', 8: u'left', 3: u'right', 1: u'down'},
#    1: {2: u'left', 9: u'up', 1: u'down', 3: u'right'},
#    2: {1: u'left', 5: u'up', 2: u'right', 8: u'down'},
#    }
#}
#
#
#subreward_function = {
#    0: {
#    0: {"A": 1, "B": 0, "C": 0},
#    1: {"A": 0, "B": 1, "C": 0},
#    2: {"A": 0, "B": 0, "C": 1},
#    },
#    1: {
#    0: {"A": 1, "B": 0, "C": 0},
#    1: {"A": 0, "B": 1, "C": 0},
#    2: {"A": 0, "B": 0, "C": 1},
#    },
#    2: {
#    0: {"A": 1, "B": 0, "C": 0},
#    1: {"A": 0, "B": 1, "C": 0},
#    2: {"A": 0, "B": 0, "C": 1},
#    },
#    3: {
#    0: {"A": 1, "B": 0, "C": 0},
#    1: {"A": 0, "B": 1, "C": 0},
#    2: {"A": 0, "B": 0, "C": 1},
#    },
#    4: {
#    0: {"A": 1, "B": 0, "C": 0},
#    1: {"A": 0, "B": 1, "C": 0},
#    2: {"A": 0, "B": 0, "C": 1},
#    },
#    5: {
#    0: {"A": 1, "B": 0, "C": 0},
#    1: {"A": 0, "B": 1, "C": 0},
#    2: {"A": 0, "B": 0, "C": 1},
#    }
#}


grid_world_size = (6, 6)
subgrid_world_size = (6, 6)

# n_rooms = 9
n_rooms = len(room_mappings)

# make it easy, have start locations be the same for each room
start_location = {r: (0,0) for r in range(n_rooms)}

# shall also have start locations be same in each sublvl

door_ids = ["A","B","C","D"]
door_coords = [(1, 2), (2, 5), (3, 1), (4, 3)]
doors = np.array(zip(door_ids, door_coords))
n_doors = len(door_coords)
#door_order = [random.sample(xrange(n_doors), n_doors) for i in range(n_rooms)]
#door_order = [[0,1,2,3]]*(n_rooms/2) + [[1,3,2,0]]*(n_rooms/2)
door_locations = {r: doors[door_order[r]] for r in range(n_rooms)}

sublvl_door_locations = {r: {door : door_coords[ii] for ii, door in enumerate(door_ids)} for r in range(n_rooms)}


n_sims = 2

alpha = 1.0
inv_temp = 5.0

rooms_args = list([room_mappings, start_location, door_locations, sublvl_mappings, subreward_function, sublvl_door_locations])

n_agents = 3

#random.seed(datetime.now())
#seed0 = random.randint(0,10000)
#seed = int(seed0)
seed = 65756
np.random.seed(seed)
rand_seeds = np.random.randint(np.iinfo(np.int32).max, size=(n_agents,n_sims))



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


num_cores = multiprocessing.cpu_count()

inputs = tqdm(range(n_sims))
results_hc = Parallel(n_jobs=num_cores)(delayed(simulate_one)(HierarchicalAgent, ii, rooms_args, seed=rand_seeds[0,ii]) for ii in inputs)

inputs = tqdm(range(n_sims))
results_ic = Parallel(n_jobs=num_cores)(delayed(simulate_one)(IndependentClusterAgent, ii, rooms_args, seed=rand_seeds[0,ii]) for ii in inputs)

inputs = tqdm(range(n_sims))
results_fl = Parallel(n_jobs=num_cores)(delayed(simulate_one)(FlatAgent, ii, rooms_args, seed=rand_seeds[0,ii]) for ii in inputs)

results = [results_hc, results_ic, results_fl]
results = pd.concat(results)

results.to_pickle("./HierarchicalRooms.pkl")