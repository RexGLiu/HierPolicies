import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from model.comp_rooms import RoomsProblem
from model.comp_rooms_agents import IndependentClusterAgent, JointClusteringAgent, FlatAgent

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import numpy as np

mappings = {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
}

room_mappings = {
    0: mappings[0],
    1: mappings[0],
    2: mappings[1],
    3: mappings[1],
    4: mappings[2],
    5: mappings[2],
}


sublvl_mappings = {0: {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
    },
    1: {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
    },
    2: {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
    },
    3: {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
    },
    4: {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
    },
    5: {
    0: {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
    1: {4: u'up', 5: u'left', 6: u'right', 7: u'down'},
    2: {1: u'left', 0: u'up', 3: u'right', 2: u'down'},
    3: {5: u'up', 4: u'left', 7: u'down', 6: u'right'},
    }
}


subreward_function = {
    0: {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 1, "B": 0, "C": 0},
    2: {"A": 1, "B": 0, "C": 0},
    3: {"A": 1, "B": 0, "C": 0},
    },
    1: {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 1, "B": 0, "C": 0},
    2: {"A": 1, "B": 0, "C": 0},
    3: {"A": 1, "B": 0, "C": 0},
    },
    2: {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 1, "B": 0, "C": 0},
    2: {"A": 1, "B": 0, "C": 0},
    3: {"A": 1, "B": 0, "C": 0},
    },
    3: {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 1, "B": 0, "C": 0},
    2: {"A": 1, "B": 0, "C": 0},
    3: {"A": 1, "B": 0, "C": 0},
    },
    4: {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 1, "B": 0, "C": 0},
    2: {"A": 1, "B": 0, "C": 0},
    3: {"A": 1, "B": 0, "C": 0},
    },
    5: {
    0: {"A": 1, "B": 0, "C": 0},
    1: {"A": 1, "B": 0, "C": 0},
    2: {"A": 1, "B": 0, "C": 0},
    3: {"A": 1, "B": 0, "C": 0},
    }
}


grid_world_size = (6, 6)

# n_rooms = 9
n_rooms = len(room_mappings)

# make it easy, have the door and start locations be the same for each room
start_location = {r: (0,0) for r in range(n_rooms)}

# make it easy, each door in sublevel is in the same spot
subdoor_locations = {r: {'A':(5, 5), 'B':(5, 0), 'C':(0, 5)} for r in range(n_rooms)}

sublvl_coords = [(1, 2), (2, 5), (3, 1), (4, 3)]
n_sublvls = len(sublvl_coords)
sublvl_order = [random.sample(xrange(n_sublvls), n_sublvls) for i in range(n_rooms)]
sublvl_locations = {r: [('A',sublvl_coords[sublvl_order[r][0]]), ('B',sublvl_coords[sublvl_order[r][1]]), ('C',sublvl_coords[sublvl_order[r][2]]), ('D',sublvl_coords[sublvl_order[r][3]])] for r in range(n_rooms)}


n_sims = 1

generate_kwargs = {
    'prunning_threshold': 10.0,
    'evaluate': False,
}

alpha = 1.0
inv_temp = 5.0


sublvl_args = [list([sublvl_locations[i], sublvl_mappings[i], subreward_function[i],
                  subdoor_locations]) for i in range(n_rooms)]

# rooms_args = list([room_mappings, sucessor_function, reward_function, start_location,
#                   door_locations, sublvl_args])

rooms_args = list([room_mappings, start_location, sublvl_args])


def sim_task(rooms_args, desc='Running Task'):

    results = []
    for ii in tqdm(range(n_sims), desc=desc):

        rooms_kwargs = dict()
        
        # task = RoomsProblem(*rooms_args, **rooms_kwargs)
        # agent = IndependentClusterAgent(task, alpha=alpha, inv_temp=inv_temp)
        # results_ic = agent.navigate_rooms(**generate_kwargs)
        # results_ic[u'Model'] = 'Independent'
        # results_ic['Iteration'] = [ii] * len(results_ic)

        # task = RoomsProblem(*rooms_args, **rooms_kwargs)
        # agent = JointClusteringAgent(task, alpha=alpha, inv_temp=inv_temp)
        # results_jc = agent.navigate_rooms(**generate_kwargs)
        # results_jc[u'Model'] = 'Joint'
        # results_jc['Iteration'] = [ii] * len(results_jc)

        task = RoomsProblem(*rooms_args, **rooms_kwargs)
        agent = FlatAgent(task, inv_temp=inv_temp)
        results_fl = agent.navigate_rooms(**generate_kwargs)
        results_fl[u'Model'] = 'Flat'
        results_fl['Iteration'] = [ii] * len(results_fl)

        # results.append(results_ic)
        # results.append(results_jc)
        results.append(results_fl)
    return pd.concat(results)


results = sim_task(rooms_args)