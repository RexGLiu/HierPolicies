#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:56:09 2020

@author: rex
"""

import itertools
import numpy as np
import numpy.matlib 

n_aa = 4
n_a = 8
actions = (u'left', u'up', u'down', u'right')

n_rooms = 8
n_upper_doors = 4
n_lower_goals = 3
n_sublvls = n_upper_doors-1

n_upper_clusters = 5
n_lower_map_clusters = 10

lower_goal_names = ("A","B","C")

assert len(lower_goal_names) == n_lower_goals
assert len(actions) == n_aa


# geometric distribution for choosing upper level clusters
odds = 0.7
init = (1.0-odds)/(1.0 - odds**n_upper_clusters)
upper_prob = init*np.array([odds**ii for ii in range(n_upper_clusters)])


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







# shuffle list of mappings again for lower levels
perm_idx = np.random.permutation(len(mappings_list))
mappings_list = mappings_list[perm_idx]

# select n_lower_clusters of these mappings
mappings = range(n_lower_map_clusters)
for ii, transition in enumerate(mappings_list):
    if ii == n_lower_map_clusters:
        break
    mappings[ii] = dict(zip(transition, actions))
    
    
# generate list of goal rewards for lower levels
lower_goal_list = range(n_lower_goals)
for ii in range(n_lower_goals):
    reward = [0]*n_lower_goals
    reward[ii] = 1
    lower_goal_list[ii] = dict(zip(lower_goal_names, reward))

lower_goal_list = np.array(lower_goal_list)
lower_goal_list = np.random.permutation(lower_goal_list)




# geometric distribution for upper context affecting sublevel mappings
odds = 0.7
upper_dist = np.array([odds**ii for ii in range(n_lower_map_clusters)])

# geometric distribution for sublevel type affecting sublevel mappings
odds = 0.9
lower_dist = np.array([odds**ii for ii in range(n_lower_map_clusters)])

# for each upper context, sublvl pair, generate probability of each mapping cluster
sublvl_map_dist = np.ones((n_lower_map_clusters, n_upper_clusters, n_sublvls))
for ii in range(n_sublvls):
    lower_dist = np.random.permutation(lower_dist)
    lower_dist_tmp = lower_dist.reshape(n_lower_map_clusters,1)
    sublvl_map_dist[:,:,ii] *= lower_dist_tmp

for ii in range(n_upper_clusters):
    upper_dist = np.random.permutation(upper_dist)
    upper_dist_tmp = upper_dist.reshape(n_lower_map_clusters,1)
    sublvl_map_dist[:,ii,:] *= upper_dist_tmp

sublvl_map_dist_norms = np.sum(sublvl_map_dist, axis=0)
sublvl_map_dist_norms = np.reshape(sublvl_map_dist_norms, (1, n_upper_clusters, n_sublvls))
sublvl_map_dist /= sublvl_map_dist_norms

sublvl_mappings = dict()
sublvl_idx = range(n_sublvls)
for ii in range(n_rooms):
    upper_cluster_type = upper_map_clusters[ii]
    
    room_sublvl_mappings = dict()
    for jj in range(n_sublvls):
        sublvl_dist = sublvl_map_dist[:,upper_cluster_type,jj]
        room_sublvl_mappings[jj] = np.random.choice(mappings,p=sublvl_dist)
        
    sublvl_mappings[ii] = room_sublvl_mappings
    
    
    
# geometric distribution for upper context affecting sublevel mappings
odds = 0.7
upper_dist = np.array([odds**ii for ii in range(n_lower_goals)])

# geometric distribution for sublevel type affecting sublevel mappings
odds = 0.9
lower_dist = np.array([odds**ii for ii in range(n_lower_goals)])

# for each upper context, sublvl pair, generate probability of each mapping cluster
sublvl_goal_dist = np.ones((n_lower_goals, n_upper_clusters, n_sublvls))
for ii in range(n_sublvls):
    lower_dist = np.random.permutation(lower_dist)
    lower_dist_tmp = lower_dist.reshape(n_lower_goals,1)
    sublvl_goal_dist[:,:,ii] *= lower_dist_tmp

for ii in range(n_upper_clusters):
    upper_dist = np.random.permutation(upper_dist)
    upper_dist_tmp = upper_dist.reshape(n_lower_goals,1)
    sublvl_goal_dist[:,ii,:] *= upper_dist_tmp

sublvl_goal_dist_norms = np.sum(sublvl_goal_dist, axis=0)
sublvl_goal_dist_norms = np.reshape(sublvl_goal_dist_norms, (1, n_upper_clusters, n_sublvls))
sublvl_goal_dist /= sublvl_goal_dist_norms

subreward_function = dict()
sublvl_idx = range(n_sublvls)
for ii in range(n_rooms):
    upper_cluster_type = upper_map_clusters[ii]
    
    room_subrewards = dict()
    for jj in range(n_sublvls):
        sublvl_dist = sublvl_goal_dist[:,upper_cluster_type,jj]
        room_subrewards[jj] = np.random.choice(lower_goal_list,p=sublvl_dist)
        
    subreward_function[ii] = room_subrewards