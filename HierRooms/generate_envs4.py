#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:56:09 2020

@author: rex
"""

import itertools
import numpy as np
import numpy.matlib 

#np.random.seed(3)

n_aa = 4
n_a = 8
actions = (u'left', u'up', u'down', u'right')

n_rooms = 6
n_upper_doors = 4
n_lower_goals = 3
n_sublvls = n_upper_doors-1

n_upper_clusters = 3
n_lower_map_clusters = 10

lower_goal_names = ("A","B","C")

assert len(lower_goal_names) == n_lower_goals
assert len(actions) == n_aa


# geometric distribution for choosing upper level clusters
odds = 0.7
init = (1.0-odds)/(1.0 - odds**n_upper_clusters)
upper_prob = init*np.array([odds**ii for ii in range(n_upper_clusters)])


# generate door orderings for upper levels
upper_door_orderings = np.array(list(itertools.permutations(range(n_upper_doors))))

# shuffle list of orderings
upper_door_perm_idx = np.random.permutation(len(upper_door_orderings))
upper_door_orderings = upper_door_orderings[upper_door_perm_idx]

# select n_upper_clusters of these orderings
door_orderings = range(n_upper_clusters)
for ii, ordering in enumerate(upper_door_orderings):
    if ii == n_upper_clusters:
        break
    door_orderings[ii] = ordering
    
door_orderings = np.array(door_orderings)

# from these n_upper_clusters orderings, randomly select orderings for each room
# according to a geometric distribution
ordering_idx = np.random.choice(n_upper_clusters,n_rooms,p=upper_prob)
door_order = door_orderings[ordering_idx]
door_order = door_order.tolist()





# generate action mappings for upper levels

# generate all possible mappings of actions to abstract actions
mappings_list = np.array(list(itertools.permutations(range(n_a),r=n_aa)))

# shuffle list of mappings
perm_idx = np.random.permutation(len(mappings_list))
mappings_list = mappings_list[perm_idx]

# select n_upper_clusters of these mappings for the upper levels
mappings = range(n_upper_clusters)
for ii, transition in enumerate(mappings_list):
    if ii == n_upper_clusters:
        break
    mappings[ii] = dict(zip(transition, actions))
mappings = np.array(mappings)
    
# from these n_upper_clusters mappings, randomly select mappings for each room
# according to a geometric distribution
upper_map_clusters = np.random.choice(n_upper_clusters,n_rooms,p=upper_prob)
mappings_selection = mappings[upper_map_clusters]
room_mappings = dict(zip(range(n_rooms),mappings_selection))





# generate list of goal rewards for lower levels
lower_goal_list = range(n_lower_goals)
for ii in range(n_lower_goals):
    reward = [0]*n_lower_goals
    reward[ii] = 1
    lower_goal_list[ii] = dict(zip(lower_goal_names, reward))

lower_goal_list = np.array(lower_goal_list)
lower_goal_list = np.array(list(itertools.permutations(lower_goal_list)))
lower_goal_list = np.random.permutation(lower_goal_list)


# sublvl rewards made independent and non repeating between diff sublvl types

# geometric distribution for sublevel type affecting sublevel goals
n_goal_permutations = len(lower_goal_list)
odds = 0.7
init = (1.0-odds)/(1.0 - odds**n_goal_permutations)
lower_dist = init*np.array([odds**ii for ii in range(n_goal_permutations)])

subreward_function = dict()
sublvl_idx = range(n_sublvls)
for ii in range(n_rooms):
    reward_idx = np.random.choice(n_goal_permutations,p=lower_dist)
    subreward_function[ii] = lower_goal_list[reward_idx]




# shuffle list of mappings again for lower levels
perm_idx = np.random.permutation(len(mappings_list))
mappings_list = mappings_list[perm_idx]

# select n_lower_clusters of these mappings
mappings = range(n_lower_map_clusters)
for ii, transition in enumerate(mappings_list):
    if ii == n_lower_map_clusters:
        break
    mappings[ii] = dict(zip(transition, actions))
    
    



# geometric distribution for sublevel type affecting sublevel mappings
odds = 1
#init = (1.0-odds)/(1.0 - odds**n_lower_map_clusters)
#lower_dist = init*np.array([odds**ii for ii in range(n_lower_map_clusters)])
lower_dist = np.zeros(n_lower_map_clusters)
lower_dist[0] = 1.

# for each upper context, sublvl pair, generate probability of each mapping cluster
sublvl_map_dist = np.ones((n_lower_map_clusters, n_upper_clusters, n_sublvls))
for ii in range(n_sublvls):
    for jj in range(n_upper_clusters):
        lower_dist = np.random.permutation(lower_dist)
        sublvl_map_dist[:,jj,ii] = lower_dist

#sublvl_map_dist_norms = np.sum(sublvl_map_dist, axis=0)
#sublvl_map_dist_norms = np.reshape(sublvl_map_dist_norms, (1, n_upper_clusters, n_sublvls))
#sublvl_map_dist /= sublvl_map_dist_norms

sublvl_mappings = dict()
sublvl_idx = range(n_sublvls)
for ii in range(n_rooms):
    upper_cluster_type = upper_map_clusters[ii]
    
    room_sublvl_mappings = dict()
    for jj in range(n_sublvls):
        sublvl_dist = sublvl_map_dist[:,upper_cluster_type,jj]
        room_sublvl_mappings[jj] = np.random.choice(mappings,p=sublvl_dist)
        
    sublvl_mappings[ii] = room_sublvl_mappings
    
    
    