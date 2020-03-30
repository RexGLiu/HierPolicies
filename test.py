import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from model.comp_rooms import RoomsProblem
from model.comp_rooms_agents import HierarchicalAgent, IndependentClusterAgent, FlatAgent#, JointClusteringAgent

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

import numpy as np

#execfile("generate_envs.py")
execfile("generate_envs2.py")



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

# make it easy, each door in sublevel is in the same spot
sublvl_door_locations = {r: {'A':(5, 5), 'B':(5, 0), 'C':(0, 5)} for r in range(n_rooms)}

door_ids = ["A","B","C","D"]
door_coords = [(1, 2), (2, 5), (3, 1), (4, 3)]
doors = np.array(zip(door_ids, door_coords))
n_doors = len(door_coords)
#door_order = [random.sample(xrange(n_doors), n_doors) for i in range(n_rooms)]
#door_order = [[0,1,2,3]]*(n_rooms/2) + [[1,3,2,0]]*(n_rooms/2)
door_locations = {r: doors[door_order[r]] for r in range(n_rooms)}

n_sims = 1

alpha = 1.0
inv_temp = 5.0

rooms_args = list([room_mappings, start_location, door_locations, sublvl_mappings, subreward_function, sublvl_door_locations])


def sim_task(rooms_args, desc='Running Task'):

    results = []
    for ii in tqdm(range(n_sims), desc=desc):
        
        # print 'Hierarchical'

        # rooms_kwargs = dict()
        
        # generate_kwargs = {
        #     'evaluate': False,
        #     }

        # task = RoomsProblem(*rooms_args, **rooms_kwargs)
        # agent = HierarchicalAgent(task, alpha=alpha, inv_temp=inv_temp)
        # results_hc = agent.navigate_rooms(**generate_kwargs)
        # results_hc[u'Model'] = 'Hierarchical'
        # results_hc['Iteration'] = [ii] * len(results_hc)


        print 'Independent'

        rooms_kwargs = dict()
        
        generate_kwargs = {
            'prunning_threshold': 1.5,
#            'evaluate': False,
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
#        print 'Flat'
#        
#        rooms_kwargs = dict()
#
#        generate_kwargs = {
##            'prunning_threshold': 10.0,
#            'evaluate': False,
#            }
#
#        task = RoomsProblem(*rooms_args, **rooms_kwargs)
#        agent = FlatAgent(task, inv_temp=inv_temp)
#        results_fl = agent.navigate_rooms(**generate_kwargs)
#        results_fl[u'Model'] = 'Flat'
#        results_fl['Iteration'] = [ii] * len(results_fl)
#
#        results.append(results_hc)
#        results.append(results_ic)
#        # results.append(results_jc)
#        results.append(results_fl)
    return pd.concat(results)


results = sim_task(rooms_args)




#sns.set_context('paper', font_scale=1.25)
#X0 = results[results['In Goal']].groupby(['Model', 'Iteration']).sum()
#from matplotlib import gridspec
#
#with sns.axes_style('ticks'):
#    cc = sns.color_palette('Dark2')
#    fig = plt.figure(figsize=(6, 3)) 
#    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
#    ax0 = plt.subplot(gs[0])
#    ax1 = plt.subplot(gs[1])
#
#    sns.distplot(X0.loc['Independent']['Step'], label='Ind.', ax=ax0, color=cc[1])
##    sns.distplot(X0.loc['Joint']['Step'], label='Joint', ax=ax0, color=cc[2])
#    sns.distplot(X0.loc['Flat']['Step'], label='Flat', ax=ax0, color=cc[0])
#    handles, labels = ax0.get_legend_handles_labels()
#    ax0.legend(handles, labels)
#    ax0.set_yticks([])
##     ax0.set_ylabel('Density')
#    ax0.set_xlim([0, ax0.get_xlim()[1] ])
#    ax0.set_xlabel('Cumulative Steps')
##     ax0.set_xticks(np.arange(0, 1501, 500))
#    
#    X1 = pd.DataFrame({
#        'Cumulative Steps Taken': np.concatenate([
#                X0.loc['Joint']['Step'].values,
#                X0.loc['Independent']['Step'].values,
#                X0.loc['Flat']['Step'].values, 
#            ]),
#        'Model': ['Joint'] * n_sims + ['Independent'] * n_sims + ['Flat'] * n_sims,
#    })
#    sns.barplot(data=X1, x='Model', y='Cumulative Steps Taken', ax=ax1, 
#                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Joint'])
#    ax1.set_ylabel('Total Steps')
#    ax1.set_xticklabels(['Flat', 'Ind.', 'Joint'])
#
#    sns.despine(offset=2)    
#    ax0.spines['left'].set_visible(False)
#
#    plt.tight_layout()
##    fig.savefig('RoomsResults.png', dpi=300)



sns.set_context('paper', font_scale=1.25)
X0 = results[results['In Goal']].groupby(['Model', 'Iteration']).sum()
from matplotlib import gridspec

with sns.axes_style('ticks'):
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(6, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    sns.distplot(X0.loc['Hierarchical']['Step'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X0.loc['Independent']['Step'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X0.loc['Flat']['Step'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlabel('Cumulative Steps')
    
    X1 = pd.DataFrame({
        'Cumulative Steps Taken': np.concatenate([
                X0.loc['Hierarchical']['Step'].values,
                X0.loc['Independent']['Step'].values,
                X0.loc['Flat']['Step'].values, 
            ]),
        'Model': ['Hierarchical'] * n_sims + ['Independent'] * n_sims + ['Flat'] * n_sims,
    })
    sns.barplot(data=X1, x='Model', y='Cumulative Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Total Steps')
    ax1.set_xticklabels(['Flat', 'Ind.', 'Hier.'])

    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)

    plt.tight_layout()























        
#        print task.get_current_room()        
#        u0 = 1
#        r0 = 3
#        u = 1
#        right = 3
#        print task.get_location(), task.get_current_lvl()
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        u = 4
#        right = 6
#        print task.get_location(), task.get_current_lvl()
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#
#        d0 = 2        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        print end_location, task.get_current_lvl()
#        
#        u = 0
#        right = 3
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        print ' '
#        print task.get_current_room()  
#        print task.get_location(), task.get_current_lvl()
#        u0 = 1
#        r0 = 3
#        u = 1
#        right = 3
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        u = 4
#        right = 6
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#
#        d0 = 2  
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        print end_location, task.get_current_lvl()
#        
#        u = 0
#        right = 3
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        print task.get_current_room()       
#        
#        
#        
#                
#        
#        print ' '
#        print task.get_current_room()  
#        print task.get_location(), task.get_current_lvl()
#        u0 = 4
#        r0 = 6
#        u = 1
#        right = 3
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        u = 4
#        right = 6
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#
#        d0 = 7
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        print end_location, task.get_current_lvl()
#        
#        u = 0
#        right = 3
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        
#        print ' '
#        print task.get_current_room()  
#        print task.get_location(), task.get_current_lvl()
#        u0 = 4
#        r0 = 6
#        u = 1
#        right = 3
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        u = 4
#        right = 6
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#
#        d0 = 7
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        print end_location, task.get_current_lvl()
#        
#        u = 0
#        right = 3
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#
#
#
#        print ' '
#        print task.get_current_room()  
#        print task.get_location(), task.get_current_lvl()
#        u0 = 0
#        r0 = 3
#        u = 1
#        right = 3
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        u = 4
#        right = 6
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#
#        d0 = 2
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        print end_location, task.get_current_lvl()
#        
#        u = 0
#        right = 3
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        
#        
#        print ' '
#        print task.get_current_room()  
#        print task.get_location(), task.get_current_lvl()
#        u0 = 0
#        r0 = 3
#        u = 1
#        right = 3
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location, task.get_current_lvl()
#        
#        
#        u = 4
#        right = 6
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#
#        d0 = 2
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        aa, end_location, goal_id, r = task.move(d0)
#        print end_location, task.get_current_lvl()
#        
#        u = 0
#        right = 3
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(u)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(right)
#        aa, end_location, goal_id, r = task.move(u)
#        print end_location, task.get_current_lvl()
#        
#        aa, end_location, goal_id, r = task.move(r0)
#        aa, end_location, goal_id, r = task.move(u0)
#        print end_location
#        print task.get_current_room()  
