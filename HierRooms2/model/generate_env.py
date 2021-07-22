#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:56:09 2020

@author: rex
"""

import numpy as np
import numpy.matlib
from itertools import permutations, product
from model.generate_env import *


def generate_mappings_set(n_a):
# we just need a permutation of the numbers 0 to 7 in each column of mappings
# s.t. no number appears twice in any row. Example below:
#
# mappings = [
#     {0: u'left', 1: u'up', 2: u'down', 3: u'right'},
#     {1: u'left', 2: u'up', 3: u'down', 4: u'right'},
#     {2: u'left', 3: u'up', 4: u'down', 5: u'right'},
#     {3: u'left', 4: u'up', 5: u'down', 6: u'right'},
#     {4: u'left', 5: u'up', 6: u'down', 7: u'right'},
#     {5: u'left', 6: u'up', 7: u'down', 0: u'right'},
#     {6: u'left', 7: u'up', 0: u'down', 1: u'right'},
#     {7: u'left', 0: u'up', 1: u'down', 2: u'right'},
# ]

    a_idx = np.random.permutation(n_a) # initialise a random permutation for first col
    mappings_set = np.array([np.concatenate((a_idx[ii:],a_idx[:ii])) for ii in range(n_a)]) # cyclically permute first col to get remaining cols
    mappings_set = mappings_set[:,np.random.permutation(n_a)]  # randomly permute ordering of cols
    
    return mappings_set

def generate_mapping_definitions(mappings_set, actions, mappings_idx):
    n_aa = len(actions)
    mapping_definitions = { ii : { mappings_set[a][idx] : actions[a]  for a in range(n_aa) } for ii, idx in enumerate(mappings_idx)}
    return mapping_definitions

def generate_sublvl_mappings(mappings_sets, actions, mappings_idx):
    n_sublvls = len(mappings_sets)
    n_rooms = len(mappings_idx[0])

    mapping_definitions = {}
    for ii in range(n_rooms):
        room_sublvl_mappings = {}
        for sublvl in range(n_sublvls):
            set_idx = mappings_idx[sublvl][ii]
            mapping = mappings_sets[sublvl][:,set_idx]
            room_sublvl_mappings[sublvl] = dict(zip(mapping, actions))
        mapping_definitions[ii] = room_sublvl_mappings
    
    return mapping_definitions

def assign_doors_to_coords(door_ids, door_coords):
    n_doors = len(door_ids)
    perm = np.random.permutation(n_doors)
    permuted_door_coords = [door_coords[ii] for ii in perm]
    doors = np.array(zip(door_ids, permuted_door_coords))
    
    return doors

def generate_door_locations(doors, door_sequences_idx, repetitions=False):
    # generate door orderings for upper levels

    n_rooms = len(door_sequences_idx)
    n_doors = len(doors)
    if repetitions:
        door_permutations = np.array(list(product(range(n_doors), repeat=n_doors)))
    else:
        door_permutations = np.array(list(permutations(range(n_doors))))

    # shuffle list of orderings
    perm = np.random.permutation(len(door_permutations))
    door_permutations = door_permutations[perm]
    
    # generate list of orderings in which door must be visited in each room
    door_orderings = [ door_permutations[ii] for ii in door_sequences_idx ]
    
    # convert orderings to door name and location
    door_locations = {r: doors[door_orderings[r]] for r in range(n_rooms)}
    
    return door_locations

def generate_sublvl_goals_list(goal_ids, repetitions=False, n_sublvls=0):
    n_goals = len(goal_ids)
    
    goal_list = range(n_goals)
    for ii in range(n_goals):
        reward = [0]*n_goals
        reward[ii] = 1
        goal_list[ii] = dict(zip(goal_ids, reward))

    goal_list = np.array(goal_list)
    if repetitions:
        assert n_sublvls > 0
        goal_list = np.array(list(product(goal_list, repeat=n_sublvls)))
    else:
        goal_list = np.array(list(permutations(goal_list)))  # each sublvl in a room will have a distinct goal
    goal_list = np.random.permutation(goal_list)

    return goal_list

def generate_sublvl_reward_fn(sublvl_goals_list, sublvl_rewards_idx):
    sublvl_reward_fn = { room : sublvl_goals_list[idx] for room, idx in enumerate(sublvl_rewards_idx) }
    return sublvl_reward_fn


def generate_room_args(actions, n_a, goal_ids, goal_coods, room_mappings_idx, door_sequences_idx, sublvl_rewards_idx, 
             sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx):
    
    # randomise order in which agent must navigate rooms
    n_rooms = len(room_mappings_idx)
    room_order = np.random.permutation(n_rooms)

    room_mappings_idx = room_mappings_idx[room_order]
    door_sequences_idx = door_sequences_idx[room_order]
    sublvl_rewards_idx = sublvl_rewards_idx[room_order]
    sublvl1_mappings_idx = sublvl1_mappings_idx[room_order]
    sublvl2_mappings_idx = sublvl2_mappings_idx[room_order]
    sublvl3_mappings_idx = sublvl3_mappings_idx[room_order]


    # generate dictionary of mappings for each room
    # to encourage transfer, we'll use same set of mappings for room and sublvls
    mappings_set = generate_mappings_set(n_a)
    sublvl1_mappings_set = mappings_set[:,np.random.permutation(n_a)]
    sublvl2_mappings_set = mappings_set[:,np.random.permutation(n_a)]
    sublvl3_mappings_set = mappings_set[:,np.random.permutation(n_a)]


    room_mappings = generate_mapping_definitions(mappings_set, actions, room_mappings_idx)

    sublvl_mappings_sets = (sublvl1_mappings_set, sublvl2_mappings_set, sublvl3_mappings_set)
    sublvl_mappings_idx = (sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx)
    sublvl_mappings = generate_sublvl_mappings(sublvl_mappings_sets, actions, sublvl_mappings_idx)


    doors = assign_doors_to_coords(goal_ids, goal_coods)
    door_locations = generate_door_locations(doors, door_sequences_idx)

    sublvl_door_locations = {r: dict(zip(goal_ids, goal_coods)) for r in range(n_rooms)}
    sublvl_goals_list = generate_sublvl_goals_list(goal_ids)
    subreward_function = generate_sublvl_reward_fn(sublvl_goals_list, sublvl_rewards_idx)
    # in a given room, sublvls will have different goals from each other

    # make it easy, have start locations be the same for each room
    start_location = {r: (0,0) for r in range(n_rooms)}
    
    return [room_mappings, start_location, door_locations, sublvl_mappings, subreward_function, sublvl_door_locations]
    
    
def generate_room_args_indep(actions, n_a, goal_ids, goal_coods, room_mappings_idx, door_sequences_idx, sublvl_rewards_idx, 
             sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx, replacement=False):
    # same as generate_room_args except sublvl mapping sets are identical to room mapping sets
    
    
    # randomise order in which agent must navigate rooms
    n_rooms = len(room_mappings_idx)
    n_sublvl = 3
    room_order = np.random.permutation(n_rooms)

    room_mappings_idx = room_mappings_idx[room_order]
    door_sequences_idx = door_sequences_idx[room_order]
    sublvl_rewards_idx = sublvl_rewards_idx[room_order]
    sublvl1_mappings_idx = sublvl1_mappings_idx[room_order]
    sublvl2_mappings_idx = sublvl2_mappings_idx[room_order]
    sublvl3_mappings_idx = sublvl3_mappings_idx[room_order]


    # generate dictionary of mappings for each room
    # to encourage transfer, we'll use same set of mappings for room and sublvls
    mappings_set = generate_mappings_set(n_a)
    sublvl1_mappings_set = mappings_set
    sublvl2_mappings_set = mappings_set
    sublvl3_mappings_set = mappings_set


    room_mappings = generate_mapping_definitions(mappings_set, actions, room_mappings_idx)

    sublvl_mappings_sets = (sublvl1_mappings_set, sublvl2_mappings_set, sublvl3_mappings_set)
    sublvl_mappings_idx = (sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx)
    sublvl_mappings = generate_sublvl_mappings(sublvl_mappings_sets, actions, sublvl_mappings_idx)


    doors = assign_doors_to_coords(goal_ids, goal_coods)
    door_locations = generate_door_locations(doors, door_sequences_idx)

    sublvl_door_locations = {r: dict(zip(goal_ids, goal_coods)) for r in range(n_rooms)}
    sublvl_goals_list = generate_sublvl_goals_list(goal_ids, replacement, n_sublvl)
    subreward_function = generate_sublvl_reward_fn(sublvl_goals_list, sublvl_rewards_idx)
    # in a given room, sublvls will have different goals from each other

    # make it easy, have start locations be the same for each room
    start_location = {r: (0,0) for r in range(n_rooms)}

    return [room_mappings, start_location, door_locations, sublvl_mappings, subreward_function, sublvl_door_locations]


def generate_room_args_indep2(actions, n_a, goal_ids, goal_coods, room_mappings_idx, door_sequences_idx, sublvl_rewards_idx, 
             sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx, replacement=True):
    # same as generate_room_args except sublvl mapping sets are identical to room mapping sets
    
    
    # randomise order in which agent must navigate rooms
    n_rooms = len(room_mappings_idx)
    n_sublvl = 3
    room_order = np.random.permutation(n_rooms)

    room_mappings_idx = room_mappings_idx[room_order]
    door_sequences_idx = door_sequences_idx[room_order]
    sublvl_rewards_idx = sublvl_rewards_idx[room_order]
    sublvl1_mappings_idx = sublvl1_mappings_idx[room_order]
    sublvl2_mappings_idx = sublvl2_mappings_idx[room_order]
    sublvl3_mappings_idx = sublvl3_mappings_idx[room_order]


    # generate dictionary of mappings for each room
    # to encourage transfer, we'll use same set of mappings for room and sublvls
    mappings_set = generate_mappings_set(n_a)
    sublvl1_mappings_set = mappings_set
    sublvl2_mappings_set = mappings_set
    sublvl3_mappings_set = mappings_set


    room_mappings = generate_mapping_definitions(mappings_set, actions, room_mappings_idx)

    sublvl_mappings_sets = (sublvl1_mappings_set, sublvl2_mappings_set, sublvl3_mappings_set)
    sublvl_mappings_idx = (sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx)
    sublvl_mappings = generate_sublvl_mappings(sublvl_mappings_sets, actions, sublvl_mappings_idx)


    doors = assign_doors_to_coords(goal_ids, goal_coods)
    door_locations = generate_door_locations(doors, door_sequences_idx, replacement)

    sublvl_door_locations = {r: dict(zip(goal_ids, goal_coods)) for r in range(n_rooms)}
    sublvl_goals_list = generate_sublvl_goals_list(goal_ids, replacement, n_sublvl)
    subreward_function = generate_sublvl_reward_fn(sublvl_goals_list, sublvl_rewards_idx)
    # in a given room, sublvls will have different goals from each other

    # make it easy, have start locations be the same for each room
    start_location = {r: (0,0) for r in range(n_rooms)}

    return [room_mappings, start_location, door_locations, sublvl_mappings, subreward_function, sublvl_door_locations]


def generate_room_args_indep3(actions, n_a, goal_ids, goal_coods, room_mappings_idx, door_sequences_idx, sublvl_rewards_idx, 
             sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx):
    # same as generate_room_args except sublvl mapping sets are identical to room mapping sets
    # no structure in door sequences but structure in subgoal sequences
    
    
    # randomise order in which agent must navigate rooms
    n_rooms = len(room_mappings_idx)
    n_sublvl = 3
    room_order = np.random.permutation(n_rooms)

    room_mappings_idx = room_mappings_idx[room_order]
    door_sequences_idx = door_sequences_idx[room_order]
    sublvl_rewards_idx = sublvl_rewards_idx[room_order]
    sublvl1_mappings_idx = sublvl1_mappings_idx[room_order]
    sublvl2_mappings_idx = sublvl2_mappings_idx[room_order]
    sublvl3_mappings_idx = sublvl3_mappings_idx[room_order]


    # generate dictionary of mappings for each room
    # to encourage transfer, we'll use same set of mappings for room and sublvls
    mappings_set = generate_mappings_set(n_a)
    sublvl1_mappings_set = mappings_set
    sublvl2_mappings_set = mappings_set
    sublvl3_mappings_set = mappings_set


    room_mappings = generate_mapping_definitions(mappings_set, actions, room_mappings_idx)

    sublvl_mappings_sets = (sublvl1_mappings_set, sublvl2_mappings_set, sublvl3_mappings_set)
    sublvl_mappings_idx = (sublvl1_mappings_idx, sublvl2_mappings_idx, sublvl3_mappings_idx)
    sublvl_mappings = generate_sublvl_mappings(sublvl_mappings_sets, actions, sublvl_mappings_idx)


    doors = assign_doors_to_coords(goal_ids, goal_coods)
    door_locations = generate_door_locations(doors, door_sequences_idx, True)

    sublvl_door_locations = {r: dict(zip(goal_ids, goal_coods)) for r in range(n_rooms)}
    sublvl_goals_list = generate_sublvl_goals_list(goal_ids, False, n_sublvl)
    subreward_function = generate_sublvl_reward_fn(sublvl_goals_list, sublvl_rewards_idx)
    # in a given room, sublvls will have different goals from each other

    # make it easy, have start locations be the same for each room
    start_location = {r: (0,0) for r in range(n_rooms)}
    
    return [room_mappings, start_location, door_locations, sublvl_mappings, subreward_function, sublvl_door_locations]    

