import numpy as np
#import matplotlib.pyplot as plt
#import time

"""
This is a variant of the grid worlds where goals are labeled and known to the agent, but their reward value
is not. This allows dissociating a movements towards a goal from raw stimulus-response associations and
can be used to force planning (both by moving the goals from trial to trial).

Reward associations are meant to be constant for each goal, but the location is meant to change from trial
to trial (here, GridWorld instance to GridWorld instance within a task)


"""


# code to make the grid world starts here!


class GridWorld(object):
    def __init__(self, grid_world_size, walls, action_map, goal_dict, start_location, 
                 state_location_key=None, n_abstract_actions=4):
        """

        :param grid_world_size: 2x2 tuple
        :param walls: list of [x, y, 'direction_of_wall'] lists
        :param action_map: dictionary of from {a: 'cardinal direction'}
        :param goal_dict: dictionary {(x, y): ('label', r)}
        :param start_location: tuple (x, y)
        :param n_abstract_actions: int
        :return:
        """
        self.start_location = start_location
        self.current_location = start_location
        self.grid_world_size = grid_world_size
        self.walls = walls

        # need to create a transition function and reward function, which pretty much define the grid world
        n_states = grid_world_size[0] * grid_world_size[1]  # assume rectangle
        if state_location_key is None:
            self.state_location_key = \
                {(x, y): (y + x * grid_world_size[1]) for y in range(grid_world_size[1]) for x in
                    range(grid_world_size[0])}
        else:
            self.state_location_key = state_location_key

        self.inverse_state_loc_key = {value: key for key, value in self.state_location_key.iteritems()}

        # define movements as change in x and y position:
        self.cardinal_direction_key = {u'up': (0, 1), u'down': (0, -1), u'left': (-1, 0), u'right': (1, 0)}
        self.abstract_action_key = {dir_: ii for ii, dir_ in enumerate(self.cardinal_direction_key.keys())}
        self.abstract_action_key[u'wait'] = -1
        
        self.inverse_abstract_action_key = {ii: dir_ for dir_, ii in self.abstract_action_key.iteritems()}

        # redefine walls pythonicly:
        wall_key = {(x, y): wall_side for x, y, wall_side in walls}
        wall_list = wall_key.keys()

        # make transition function (usable by the agent)!
        # transition function: takes in state, abstract action, state' and returns probability
        self.transition_function = np.zeros((n_states, n_abstract_actions, n_states), dtype=float)
        for s in range(n_states):

            x, y = self.inverse_state_loc_key[s]

            # cycle through movement, check for both walls and for
            for movement, (dx, dy) in self.cardinal_direction_key.iteritems():
                aa = self.abstract_action_key[movement]

                # check if the movement stays on the grid
                if (x + dx, y + dy) not in self.state_location_key.keys():
                    self.transition_function[s, aa, s] = 1

                elif (x, y) in wall_list:
                    # check if the movement if blocked by a wall
                    if wall_key[(x, y)] == movement:
                        self.transition_function[s, aa, s] = 1
                    else:
                        sp = self.state_location_key[(x + dx, y + dy)]
                        self.transition_function[s, aa, sp] = 1
                else:
                    sp = self.state_location_key[(x + dx, y + dy)]
                    self.transition_function[s, aa, sp] = 1

        # set up goals
        self.goal_dictionary = goal_dict
        self.goal_locations = {loc: label for loc, (label, _) in goal_dict.iteritems()}
        self.goal_values = {label: r for _, (label, r) in goal_dict.iteritems()}
        
        self.goals = set([g for g, _ in goal_dict.itervalues()])


        # make the goal states self absorbing!!
        for loc in self.goal_locations.iterkeys():
            s = self.state_location_key[loc]
            self.transition_function[s, :, :] = 0.0
            self.transition_function[s, :, s] = 1.0

        # store the action map
        self.action_map = {int(key): value for key, value in action_map.iteritems()}
        
        self.n_primitive_actions = len(self.action_map.keys())

        # define a successor function in terms of key-press for game interactions
        # successor function: takes in location (x, y) and action (button press) and returns successor location (x, y)
        self.successor_function = dict()
        for s in range(n_states):
            x, y = self.inverse_state_loc_key[s]

            # this loops through keys associated with a movement (valid key-presses only)
            for key_press, movement in self.action_map.iteritems():
                dx, dy = self.cardinal_direction_key[movement]

                if (x + dx, y + dy) not in self.state_location_key.keys():
                    self.successor_function[((x, y), key_press)] = (x, y)

                # check the walls for valid movements
                elif (x, y) in wall_list:
                    if wall_key[(x, y)] == movement:
                        self.successor_function[((x, y), key_press)] = (x, y)
                    else:
                        self.successor_function[((x, y), key_press)] = (x + dx, y + dy)
                else:
                    self.successor_function[((x, y), key_press)] = (x + dx, y + dy)

        # store keys used in the task for lookup value
        self.keys_used = [key for key in self.action_map.iterkeys()]

        # store walls
        self.wall_key = wall_key

    def reset(self, start_location = None):
        if start_location is None:
            self.current_location = self.start_location
        else:
            self.current_location = start_location

    def move(self, key_press):
        """
        :param key_press: int key-press
        :return:
        """
        if key_press in self.keys_used:
            new_location = self.successor_function[self.current_location, key_press]
            # get the abstract action number
            aa = self.action_map[key_press]
        else:
            new_location = self.current_location
            aa = u'wait'

        # update the current location before returning
        self.current_location = new_location

        # goal check, and return goal id + reward
        if self.goal_check():
            goal_id, r = self.goal_probe()
            return aa, new_location, goal_id, r

        return aa, new_location, None, None

    def goal_check(self):
        if self.current_location in self.goal_dictionary.keys():
            return True
        return False

    def goal_probe(self):
        return self.goal_dictionary[self.current_location]

    def get_location(self):
        return self.current_location

    def get_goal_locations(self):
        return self.goal_locations
    
    def get_transition_function(self):
        return self.transition_function


class Task(object):
    pass


class Room(Task):
    
    def __init__(self, room_mapping,
                 # list_start_locations,
                 door_locations,
                 list_sublvl_mappings,
                 list_subreward_function,
                 sublvl_door_locations,
                 context,
                 grid_world_size=(6, 6),
                 subgrid_world_size=(6, 6),
                 n_abstract_actions=4,
                 primitive_actions=(72, 74, 75, 76, 65, 83, 68, 70),
                 # list_walls=None
                 ):

        # create a state location key
        self.state_location_key = {
            (x, y): (y + x * grid_world_size[1])
            for y in range(grid_world_size[1])
            for x in range(grid_world_size[0])
            }
        self.sublvl_state_location_key = {
            (x, y): (y + x * subgrid_world_size[1])
            for y in range(subgrid_world_size[1])
            for x in range(subgrid_world_size[0])
            }
        
        walls = []
        start_loc = (0,0)
        
        self.context = context
        self.n_sublvls = len(list_sublvl_mappings)
        self.subrooms = dict()
        
        # room is composed of 'subrooms': upper rooms alternating with sublevels
        # goal in each 'upper' subroom changes to reflect correct door agent should be targetting next
        for sublvl in range(self.n_sublvls):
            
            goal_dict = {l: (g, 0) for g, l in door_locations}
            g, l = door_locations[sublvl]
            goal_dict[l] = (g, 1)
            self.subrooms[2*sublvl] = GridWorld(grid_world_size, walls, room_mapping, goal_dict,
                                               start_loc, state_location_key=self.state_location_key)
            
            subreward_function = list_subreward_function[sublvl]
            goal_dict = {l: (g, subreward_function[g]) for g, l in sublvl_door_locations.iteritems()}
            self.subrooms[2*sublvl+1] = GridWorld(grid_world_size, walls, list_sublvl_mappings[sublvl], goal_dict,
                                                 start_loc, state_location_key=self.sublvl_state_location_key)

        # instantiate final 'upper' subroom
        goal_dict = {l: (g, 0) for g, l in door_locations[:-1]}
        g, l = door_locations[-1]
        goal_dict[l] = (g, 1)
        self.subrooms[2*self.n_sublvls] = GridWorld(grid_world_size, walls, room_mapping, goal_dict,
                                                   start_loc, state_location_key=self.state_location_key)
        
        self.current_subroom = self.subrooms[0]
        self.current_subroom_number = 0
        self.upper_room_location = (0,0)
        self.n_subrooms = len(self.subrooms)
        self.current_lvl = 0
            # 0 is upper room; otherwise, this indicates current sublvl
        
        self.abstract_action_key = self.current_subroom.abstract_action_key
        
    def move(self, action):

        aa, curr_location, subgoal_id, r = self.current_subroom.move(action)
        
        if subgoal_id is not None: # at goal, so move to next subroom/room 
            if r == 0:
                # wrong goal, return to beginning
                self.current_subroom_number = None
                self.current_subroom = None
            else:
                # correct goal
                self.current_subroom_number += 1
                
                if self.current_subroom_number % 2 == 0:
                    self.current_lvl = 0
                else:
                    self.current_lvl = self.current_subroom_number / 2 + 1
                
                if self.current_subroom_number < self.n_subrooms:
                    self.current_subroom = self.subrooms[self.current_subroom_number]
                    self.abstract_action_key = self.current_subroom.abstract_action_key
                
                    if self.current_lvl == 0:
                        # if returned to upper room
                        self.current_subroom.reset(self.upper_room_location)
                    else:    
                        self.current_subroom.reset()
                        
        elif self.current_lvl == 0:
            # check if in upper room
            self.upper_room_location = curr_location
        
        return aa, curr_location, subgoal_id, r
    
    def reset_room(self):
        self.upper_room_location = (0,0)
        self.current_subroom_number = 0
        self.current_subroom = self.subrooms[0]
        self.current_subroom.reset()
        self.current_lvl = 0

    def goal_check(self):
        return self.current_subroom.goal_check()
    
    def goal_probe(self):
        return self.current_subroom.goal_probe()
    
    def get_goal_locations(self):
        assert type(self.current_subroom) is GridWorld
        return self.current_subroom.get_goal_locations()

    def get_location(self):
        return self.current_subroom.get_location()
    
    def get_lvl(self):
        return self.current_lvl
    
    def get_transition_function(self):
        return self.current_subroom.get_transition_function()
    
    def check_restart(self):
        if self.current_subroom_number is None or self.current_subroom is None:
            assert self.current_subroom_number is None
            assert self.current_subroom is None
            return True
        
        return False
    
    def end_check(self):
        return self.current_subroom_number == self.n_subrooms
    
    def get_door_order(self):
        # returns order of current door agent is heading towards in upper room
        return self.current_subroom_number / 2
    
    def get_current_gridworld(self):
        return self.current_subroom
    
    def get_action_map(self):
        return self.current_subroom.action_map
    
    def get_walls(self):
        assert type(self.current_subroom) is GridWorld
        return self.current_subroom.walls
    

        
class RoomsProblem(Task):
    def __init__(self, room_mappings, 
                 list_start_locations,
                 list_door_locations,
                 list_sublvl_mappings,
                 list_subreward_function,
                 list_sublvl_door_locations,
                 grid_world_size=(6, 6),
                 subgrid_world_size=(6, 6),
                 n_abstract_actions=4,
                 primitive_actions=(72, 74, 75, 76, 65, 83, 68, 70),
                 # list_walls=None
                 ):

        self.rooms = dict()
        for r in range(len(room_mappings)):
            self.rooms[r] = Room(room_mappings[r], list_door_locations[r], list_sublvl_mappings[r], 
                                      list_subreward_function[r], list_sublvl_door_locations[r], r)

        self.current_room_number = 0
        self.current_room = self.rooms[0]
        self.n_rooms = len(self.rooms)
        self.rooms[None] = None  # augment rooms with end state
        
        self.n_doors = len(list_door_locations[0])
        self.sublvl_ids = [id for id, _ in list_door_locations[0]]
        self.n_sublvls = self.n_doors-1
        self.subgoal_ids = list_sublvl_door_locations[0].keys()
        self.n_sublvl_doors = len(self.subgoal_ids)
        
        assert self.n_sublvls < self.n_doors

        self.trial_number = 1  # number of rooms the agent has visited
        self.n_abstract_actions = n_abstract_actions
        self.n_primitive_actions = len(primitive_actions)
        self.primitive_actions = primitive_actions
        self.abstract_action_key = self.current_room.abstract_action_key
        
        self.upper_goal_index = {g: ii for ii, g in enumerate(self.current_room.subrooms[1].goals)}
        self.sublvl_goal_index = {g: ii for ii, g in enumerate(self.current_room.subrooms[1].goals)}


    def move(self, action):

        aa, new_location, goal_id, r = self.current_room.move(action)
        if goal_id is not None:

            if self.current_room.check_restart():
                next_room = 0
                self.current_room = self.rooms[next_room]
                self.current_room_number = next_room
                self.reset_room()
                self.trial_number += 1
                self.abstract_action_key = self.current_room.abstract_action_key

            elif self.current_room.end_check():
                next_room = self.current_room_number+1
                
                if next_room < self.n_rooms:
                    self.current_room = self.rooms[next_room]
                    self.current_room_number = next_room
                    self.reset_room()
                    self.trial_number += 1
                    self.abstract_action_key = self.current_room.abstract_action_key
                else:
                    self.current_room = None
                    self.current_room_number = None

        return aa, new_location, goal_id, r

    def reset_room(self):
        assert type(self.current_room) is Room
        self.current_room.reset_room()

    def end_check(self):
        return self.current_room_number is None

    def get_current_room(self):
        return self.current_room_number
    
    def get_current_lvl(self):
        return self.current_room.get_lvl()

    def get_current_context(self):
        return self.current_room_number*self.n_doors + self.get_current_lvl()
    
    def get_transition_function(self):
        return self.current_room.get_transition_function()

    def get_location(self):
        return self.current_room.get_location()

    def get_trial_number(self):
        return self.trial_number

    def get_current_gridworld(self):
        return self.current_room.get_current_gridworld()

    def get_goal_locations(self):
        assert type(self.current_room) is Room
        return self.current_room.get_goal_locations()

    def get_walls(self):
        assert type(self.current_room) is Room
        return self.current_room.get_walls()

    def get_action_map(self):
        assert type(self.current_room) is Room
        return self.current_room.get_action_map()

    def get_goal_values(self):
        goal_values = np.zeros(self.n_goals)
        for g, idx in self.goal_index.iteritems():
            goal_values[idx] = self.current_room.goal_values[g]
        return goal_values

    def get_goal_index(self, lvl, goal):
        if lvl == 0:
            return self.upper_goal_index[goal]
        else:
            return self.sublvl_goal_index[goal]
    
    def get_n_doors(self):
        return self.n_doors
    
    def get_n_sublvls(self):
        return self.n_sublvls

    def get_n_sublvl_doors(self):
        return self.n_sublvl_doors
    
    def get_door_order(self):
        return self.current_room.get_door_order()

    def get_n_primitive_actions(self):
        return self.n_primitive_actions
    
    def get_n_abstract_actions(self):
        return self.n_abstract_actions
    
    def get_mapping_function(self, aa):
        mapping = np.zeros((self.n_primitive_actions,
                            self.n_abstract_actions), dtype=float)
        action_map = self.current_room.get_action_map()
        for a, dir_ in action_map.iteritems():
            aa0 = self.current_room.abstract_action_key[dir_]
            mapping[a, aa0] = 1

        return np.squeeze(mapping[:, aa])


def compute_task_mutual_info(room_mappings, 
                 list_start_locations,
                 list_door_locations,
                 list_sublvl_mappings,
                 list_subreward_function,
                 list_sublvl_door_locations):
    n_rooms = len(room_mappings)
    n_sublvls = len(list_sublvl_mappings[0])
    n_doors = len(list_door_locations[0])
    
    possible_sublvl_rewards = list_subreward_function[0]
    
    task_mutual_info = {}
    
    # compute mutual info between sublvl goals and mappings
    sublvl_mappings = np.array([list_sublvl_mappings[ii][jj] for ii in range(n_rooms) for jj in range(n_sublvls)])
    sublvl_mappings_set = []
    for mappings in sublvl_mappings:
        if mappings not in sublvl_mappings_set:
            sublvl_mappings_set.append(mappings)

    sublvl_mappings_idx = np.zeros( sublvl_mappings.size )    
    for ii, mappings in enumerate(sublvl_mappings_set):
        sublvl_mappings_idx[sublvl_mappings == mappings] = ii
        

    sublvl_rewards = np.array([list_subreward_function[ii][jj] for ii in range(n_rooms) for jj in range(n_sublvls)])
    
    sublvl_rewards_idx = np.zeros( sublvl_rewards.size )
    for ii, reward_fn in enumerate(possible_sublvl_rewards):
        sublvl_rewards_idx[sublvl_rewards == reward_fn] = ii
        
    task_mutual_info['sublvl'] = sequential_mutual_info(sublvl_mappings_idx, sublvl_rewards_idx)
    
    
    # compute mutual info between upper room mappings and door sequences
    room_mappings_set = []
    for mappings in room_mappings.values():
        if mappings not in room_mappings_set:
            room_mappings_set.append(mappings)
        
    room_mappings_idx = np.zeros(n_rooms)
    for ii, mappings in enumerate(room_mappings_set):
        for jj in range(n_rooms):
            if room_mappings[jj] == mappings:
                room_mappings_idx[jj] = ii

    list_door_seq = np.array([ [ door[0] for door in list_door_locations[seq_n] ] for seq_n in range(n_rooms) ])
    door_seq_set = np.unique(list_door_seq, axis=0)
    
    door_seq_idx = np.zeros( list_door_seq.shape )
    for ii, door_seq in enumerate(door_seq_set):
        seq_comparison = (list_door_seq == door_seq)
        seq_comparison = np.cumprod(seq_comparison, axis=1)
        door_seq_idx[seq_comparison == 1] = ii
    door_seq_idx = door_seq_idx.transpose()

    door_seq_mutual_info = np.zeros( door_seq_idx.shape )
    for seq in range(n_doors):
        door_seq_mutual_info[seq] = sequential_mutual_info(room_mappings_idx, door_seq_idx[seq])
    
    task_mutual_info['upper room'] = door_seq_mutual_info



    # compute mutual info between doors x historical sequences of upper mappings + doors
    list_seq = np.concatenate((room_mappings_idx.reshape((n_rooms,1)),list_door_seq), axis=1)
    seq_set = np.unique(list_seq, axis=0)
    
    seq_idx = np.zeros( list_seq.shape )
    for ii, seq in enumerate(seq_set):
        seq_comparison = (list_seq == seq)
        seq_comparison = np.cumprod(seq_comparison, axis=1)
        seq_idx[seq_comparison == 1] = ii
    seq_idx = seq_idx.transpose()

    seq_mutual_info = np.zeros( door_seq_idx.shape )
    for seq in range(n_doors):
        seq_mutual_info[seq] = sequential_mutual_info(seq_idx[seq], seq_idx[seq+1])
    
    task_mutual_info['upper sequences'] = door_seq_mutual_info
    
    
    # cumulative info gain of sequences
    task_mutual_info['upper sequences cumulative info'] = compute_cum_info(seq_idx, normalise=False)

    # normalised cumulative info gain of sequences
    task_mutual_info['upper sequences normalised cumulative info'] = compute_cum_info(seq_idx, normalise=True)
    print task_mutual_info['upper sequences normalised cumulative info'][-1]
    
    # conditional cumulative info gain of sequences
    task_mutual_info['upper sequences conditional cum info'] = conditional_cum_info(seq_idx, normalise=False)

    # normalised conditional cumulative info gain of sequences
    task_mutual_info['upper sequences normalised conditional cum info'] = conditional_cum_info(seq_idx, normalise=True)
    

    # compute mutual info between upper room mappings and sublvl goal sequences
    sublvl_rewards = np.array([ list_subreward_function[room_num][:n_sublvls] for room_num in range(n_rooms) ])
    sublvl_rewards_labelled = np.zeros( sublvl_rewards.shape )
    for ii, reward_fn in enumerate(possible_sublvl_rewards):
        sublvl_rewards_labelled[ sublvl_rewards == reward_fn ] = ii
    sublvl_rewards_set = np.unique(sublvl_rewards_labelled, axis=0)
    
    sublvl_rewards_idx = np.zeros( sublvl_rewards.shape )
    for ii, reward_seq in enumerate(sublvl_rewards_set):
        seq_comparison = (sublvl_rewards_labelled == reward_seq)
        seq_comparison = np.cumprod(seq_comparison, axis=1)
        sublvl_rewards_idx[seq_comparison == 1] = ii
    sublvl_rewards_idx = sublvl_rewards_idx.transpose()

    goal_seq_mutual_info = np.zeros( sublvl_rewards_idx.shape )
    for seq in range(n_sublvls):
        goal_seq_mutual_info[seq] = sequential_mutual_info(room_mappings_idx, sublvl_rewards_idx[seq])

    task_mutual_info['upper mapping x sublvl rewards'] = goal_seq_mutual_info
    
    
    # compute mutual info between sublvl goals and mappings
    sublvl_mutual_info = np.zeros( sublvl_rewards_idx.shape )
    for seq in range(n_sublvls):
        sublvl_mappings_seq = sublvl_mappings_idx[np.arange(len(sublvl_mappings_idx)) % n_sublvls == seq]
        sublvl_mutual_info[seq] = sequential_mutual_info(sublvl_mappings_seq, sublvl_rewards_idx[seq])
    task_mutual_info['same sublvl mapping x goal'] = sublvl_mutual_info


    # compute mutual info between upper room doors and mappings
    door_idx = np.zeros( list_door_seq.shape )
    for ii, door in enumerate(door_seq_set[0]):
        seq_comparison = (list_door_seq == door)
        door_idx[seq_comparison == 1] = ii
    door_idx = door_idx.transpose()

    door_mutual_info = np.zeros( door_idx.shape )
    for seq in range(n_doors):
        door_mutual_info[seq] = sequential_mutual_info(room_mappings_idx, door_idx[seq])
    task_mutual_info['upper mapping x individual door'] = door_mutual_info
    
    
    # mutual info between set of all mappings and individual doors or sublvl goals
    complete_mappings_set = room_mappings_set
    for mappings in sublvl_mappings_set:
        if mappings not in complete_mappings_set:
            complete_mappings_set.append(mappings)

    complete_mappings = []
    for ii in range(n_rooms):
        complete_mappings += [room_mappings[ii]] + [list_sublvl_mappings[ii][jj] for jj in range(n_sublvls)]

    complete_mappings_idx = np.zeros(n_rooms*(n_sublvls+1))
    for ii, mappings in enumerate(complete_mappings_set):
        for jj in range(n_rooms*(n_sublvls+1)):
            if complete_mappings[jj] == mappings:
                complete_mappings_idx[jj] = ii
    
    sublvl_rewards_idx = sublvl_rewards_idx.transpose().flatten()
    
    seq_subgoal_mutual_info, seq_doors_mutual_info = sequential_mutual_info2(complete_mappings_idx, sublvl_rewards_idx, door_seq_idx)

    task_mutual_info['mappings x individual door'] = seq_doors_mutual_info
    task_mutual_info['mappings x sublvl goal'] = seq_subgoal_mutual_info
    
    return task_mutual_info


def sequential_mutual_info(X, Y):
    # inputs: X and Y are 1D arrays
    # output: 1D array where i'th element gives mutual info between X[:i+1] and Y[:i+1]
    
    seq_len = len(X)
    seq_array = np.arange(1,seq_len+1)

    X_sample_space = np.array(list(set(X)),ndmin=2)
    Y_sample_space = np.array(list(set(Y)),ndmin=2)
    
    X_sample_space_size = X_sample_space.size
    Y_sample_space_size = Y_sample_space.size

    # transpose the sample space arrays for future computations
    X_sample_space = X_sample_space.reshape((X_sample_space_size,1))
    Y_sample_space = Y_sample_space.reshape((Y_sample_space_size,1))
    
    # convert X and Y to 2d arrays to facilitate boolean comparisons w sample_space arrays
    X = np.array(X,ndmin=2)
    Y = np.array(Y,ndmin=2)
    
    # element (ii,jj) in 2d event arrays indicate whether outcome ii in sample_space 
    # occurred at jj-th position in random sequences X and Y
    X_events = (X == X_sample_space)
    Y_events = (Y == Y_sample_space)

    array_shape = X_events.shape
    X_events = X_events.reshape((array_shape[0],1,array_shape[1]))

    array_shape = Y_events.shape
    Y_events = Y_events.reshape((1,array_shape[0],array_shape[1]))


    P_X = np.cumsum( X_events, axis=2, dtype=float ) / seq_array
    P_Y = np.cumsum( Y_events, axis=2, dtype=float ) / seq_array
    
    XY_conjunctions = np.logical_and(X_events, Y_events)
    P_XY = np.cumsum( XY_conjunctions, axis=2, dtype=float ) / seq_array
    P_XY_copy = np.array(P_XY)
    
    # set 0 probabilities to 1 so that log prob evaluates to 0
    P_X[P_X == 0] = 1.
    P_Y[P_Y == 0] = 1.
    P_XY[P_XY == 0] = 1.
    
    seq_mutual_info = P_XY_copy * (np.log(P_XY) - np.log(P_X) - np.log(P_Y))
    seq_mutual_info = np.sum(np.sum(seq_mutual_info, axis=0), axis=0)
    seq_mutual_info = seq_mutual_info.reshape((1,seq_len))
    
    return seq_mutual_info


def sequential_mutual_info2(mappings, subgoals, doors):
    # used to compute sequential mutual info between goals or doors and environment-wide mappings
    # doors has dims (n_doors, n_rooms)
    # mappings has dims n_rooms * (n_sublvls + 1)
    # subgoals has dims n_rooms * n_sublvls
    n_doors, n_rooms = doors.shape
    len_mappings = mappings.size
    n_sublvls = len_mappings/n_rooms - 1
    
    mappings_sample_space = np.array(list(set(mappings)),ndmin=2)
    subgoals_sample_space = np.array(list(set(subgoals)),ndmin=2)
    doors_sample_space = np.array(list(set(doors.flatten())),ndmin=2)
    
    doors_sample_space_size = doors_sample_space.size
    
    # transpose the sample space arrays for future computations
    mappings_sample_space = mappings_sample_space.transpose()
    subgoals_sample_space = subgoals_sample_space.transpose()
    doors_sample_space = doors_sample_space.reshape((doors_sample_space_size,1,1))
    
    # convert inputs to multi-dim arrays to facilitate boolean comparisons w sample_space arrays
    mappings = np.array(mappings,ndmin=2)
    subgoals = np.array(subgoals,ndmin=2)
    doors = np.array(doors,ndmin=3)
    
    # element (ii,jj) in 2d event arrays indicate whether outcome ii in sample_space 
    # occurred at jj-th position in random sequences
    mappings_events = (mappings == mappings_sample_space)
    subgoals_events = (subgoals == subgoals_sample_space)
    doors_events = (doors == doors_sample_space)
    
    array_shape = mappings_events.shape
    mappings_events = mappings_events.reshape((1,array_shape[0],array_shape[1]))
    
    array_shape = subgoals_events.shape
    subgoals_events = subgoals_events.reshape((array_shape[0],1,array_shape[1]))

    seq_array = np.arange(1,mappings_events.shape[-1]+1)
    P_mappings = np.cumsum( mappings_events, axis=2, dtype=float ) / seq_array
    
    seq_array = np.arange(1,subgoals_events.shape[-1]+1)
    P_subgoals = np.cumsum( subgoals_events, axis=2, dtype=float ) / seq_array

    seq_array = np.arange(1,doors_events.shape[-1]+1)
    P_doors = np.cumsum( doors_events, axis=2, dtype=float ) / seq_array
    
    # set 0 probabilities to 1 so that log prob evaluates to 0
    P_mappings[P_mappings == 0] = 1.
    P_subgoals[P_subgoals == 0] = 1.
    P_doors[P_doors == 0] = 1.

    
    # compute mutual info for subgoals
    P_mappings_subset = P_mappings[:,:, np.arange(len_mappings) % (n_sublvls+1) != 0 ]
    mappings_subset = mappings_events[:,:, np.arange(len_mappings) % (n_sublvls+1) != 0 ]

    seq_array = np.arange(1,subgoals_events.shape[-1]+1)
    mappings_subgoals_conjunctions = np.logical_and(mappings_subset, subgoals_events)
    P_mappings_subgoals = np.cumsum( mappings_subgoals_conjunctions, axis=2, dtype=float ) / seq_array
    
    P_mappings_subgoals_copy = np.array(P_mappings_subgoals)
    P_mappings_subgoals[P_mappings_subgoals == 0] = 1.
    
    seq_subgoal_mutual_info = P_mappings_subgoals_copy * (np.log(P_mappings_subgoals) - np.log(P_mappings_subset) - np.log(P_subgoals))
    seq_subgoal_mutual_info = np.sum(np.sum(seq_subgoal_mutual_info, axis=0), axis=0)
    seq_subgoal_mutual_info = seq_subgoal_mutual_info.reshape((1,subgoals_events.shape[-1]))
    
    
    # compute mutual info for doors
    P_mappings_subset = P_mappings[:,:, np.arange(len_mappings) % (n_sublvls+1) == 0 ]
    mappings_subset = mappings_events[:,:, np.arange(len_mappings) % (n_sublvls+1) == 0 ]
    
    
    seq_doors_mutual_info = np.zeros((n_doors, n_rooms))
    for ii in range(n_doors):
        doors_events_set = doors_events[:,ii,:]
        doors_events_set_shape = doors_events_set.shape
        doors_events_set = doors_events_set.reshape(doors_events_set_shape[0],1,doors_events_set_shape[1])
        
        mappings_doors_conjunctions = np.logical_and(mappings_subset, doors_events_set)
        seq_array = np.arange(1,n_rooms+1)
        P_mappings_doors = np.cumsum( mappings_doors_conjunctions, axis=2, dtype=float ) / seq_array
    
        P_mappings_doors_copy = np.array(P_mappings_doors)
        P_mappings_doors[P_mappings_doors == 0] = 1.
        
        P_doors_set = P_doors[:,ii,:].reshape(doors_events_set_shape[0],1,doors_events_set_shape[1])
        
        mutual_info = P_mappings_doors_copy * (np.log(P_mappings_doors) - np.log(P_mappings_subset) - np.log(P_doors_set))
        seq_doors_mutual_info[ii] = np.sum(np.sum(mutual_info, axis=0), axis=0)
    
    return seq_subgoal_mutual_info, seq_doors_mutual_info


def compute_cum_info(list_seq, normalise=True):
    seq_len, n_seq = list_seq.shape

    list_unique = np.unique(list_seq, axis=1)
    n_unique = list_unique.shape[1]
    
    unique_info_gain = np.zeros( list_unique.shape )
    
    # for each possible subsequence, compute its log joint prob
    for sub_len in range(1,seq_len+1):
        list_subseq = list_unique[:sub_len]
        unique_subseq = np.unique(list_subseq, axis=1)
        
        for subseq in unique_subseq.T:
            subseq_idx = ( list_subseq == subseq.reshape((sub_len, 1))).all(axis=0)
            n_sub = np.sum( subseq_idx )

            # compute info content: log joint prob of subsequence
            unique_info_gain[sub_len-1, subseq_idx ] = -np.log2( float(n_sub)/n_unique )
    
    conditional_info = np.zeros((seq_len, n_seq))
    for ii in range(n_unique):
        unique_idx = ( list_seq == list_unique[:,ii].reshape((seq_len,1)) )
        unique_idx = unique_idx.all(axis=0)
        conditional_info[ :, unique_idx ] = unique_info_gain[:,ii].reshape( (seq_len, 1) )

    if normalise: 
        if n_unique > 1:
            conditional_info /= conditional_info[-1,0]
        else:
            conditional_info[:] = 1

    return conditional_info


def conditional_cum_info(list_seq, normalise=True):
    conditions_set = np.unique(list_seq[0])
    seq_len, n_seq = list_seq.shape
    
    conditional_info = np.zeros((seq_len-1, n_seq))
    
    for cond in conditions_set:
        seq_idx = (list_seq[0] == cond)
        seq_subset = list_seq[ 1:, seq_idx ]
        
        conditional_info[:,seq_idx] = compute_cum_info(seq_subset, normalise)

    return conditional_info


# def compute_cum_info(list_seq, normalise=True):
#     cum_frac_info = compute_info_gain(list_seq)

#     if normalise:
#         normalisation = np.sum(cum_frac_info[:,0])
#         if normalisation > 0:
#             cum_frac_info = cum_frac_info/normalisation

#     cum_frac_info = np.cumsum(cum_frac_info,axis=0)
#     return cum_frac_info


# def compute_info_gain(list_seq):
#     seq_len, n_seq = list_seq.shape

#     unique_seq = np.unique(list_seq, axis=1)
#     n_unique = unique_seq.shape[1]
#     prob_dict = { jj: np.sum(unique_seq[0] == jj) / float(n_unique) for jj in unique_seq[0] }

#     seq_info_gain = np.zeros( list_seq.shape )
#     for jj, prob in prob_dict.iteritems():
#         seq_info_gain[ 0, list_seq[0] == jj ] = -np.log2(prob)
    
#     if seq_len > 1:
#         for jj in unique_seq[0]:
#             seq_idx = (list_seq[0] == jj)
#             list_seq_subset = list_seq[ 1:, seq_idx ]
#             seq_info_gain[ 1:, seq_idx ] = compute_info_gain(list_seq_subset)
    
#     return seq_info_gain







