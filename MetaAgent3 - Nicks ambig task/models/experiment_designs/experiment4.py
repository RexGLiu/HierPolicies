import numpy as np
import copy
from models.misc import randomize_context_order_with_autocorrelation

# define the task parameters
grid_world_size = (6, 6)
action_map_0 = {4: u'up', 5: u'left', 6: u'right', 7: u'down'}
action_map_1 = {0: u'up', 1: u'left', 2: u'right', 3: u'down'}
goal_set_labels = [['A','D','G'], ['B','E','H'], ['C','F','I']]
goal_0 = [{'A': 1.0, 'D': 0.0, 'G': 0.0}, {'B': 1.0, 'E': 0.0, 'H': 0.0}, {'C': 1.0, 'F': 0.0, 'I': 0.0}]
goal_1 = [{'A': 0.0, 'D': 1.0, 'G': 0.0}, {'B': 0.0, 'E': 1.0, 'H': 0.0}, {'C': 0.0, 'F': 1.0, 'I': 0.0}]
goal_2 = [{'A': 0.0, 'D': 0.0, 'G': 1.0}, {'B': 0.0, 'E': 0.0, 'H': 1.0}, {'C': 0.0, 'F': 0.0, 'I': 1.0}]


def rand_goal_locations():
    q_size = grid_world_size[0] / 2
    q1 = (np.random.randint(q_size) + q_size, np.random.randint(q_size) + q_size)
    q2 = (np.random.randint(q_size), np.random.randint(q_size) + q_size)
    q3 = (np.random.randint(q_size), np.random.randint(q_size))
    q4 = (np.random.randint(q_size) + q_size, np.random.randint(q_size))

    goal_locations = [q1, q2, q3, q4]
    np.random.shuffle(goal_locations)

    return goal_locations


def rand_init_loc(min_manhattan_dist=3):
    t = 0
    goal_locations = rand_goal_locations()
    while True:
        loc = (np.random.randint(grid_world_size[0]), np.random.randint(grid_world_size[1]))
        min_dist = grid_world_size[0]

        for g_loc in goal_locations:
            d = np.abs(g_loc[0] - loc[0]) + np.abs(g_loc[1] - loc[1])
            min_dist = np.min([min_dist, d])

        if min_dist >= min_manhattan_dist:
            return goal_locations, loc
        t += 1

        if t > 100:
            goal_locations = rand_goal_locations()
            t = 0



def make_trial(context=0, subtask_idx=0):
    # select a random wall generator to generate walls
    # wall_generator = [
    #     make_wall_set_a, make_wall_set_b, make_wall_set_c,
    # ][np.random.randint(3)]
    # walls = wall_generator()

    #
    context = context % 3
    
    goal_locations, start_location = rand_init_loc()

    definitions_dict = {
        0:  (goal_0[subtask_idx], action_map_0),
        1:  (goal_1[subtask_idx], action_map_0),
        2:  (goal_2[subtask_idx], action_map_1),
    }
    goal_set, action_map = definitions_dict[context]
    
    goal_labels = goal_set_labels[subtask_idx]

    goal_dict = {
        goal_locations[0]: (goal_labels[0], goal_set[goal_labels[0]]),
        goal_locations[1]: (goal_labels[1], goal_set[goal_labels[1]]),
        goal_locations[2]: (goal_labels[2], goal_set[goal_labels[2]]),
    }

    return start_location, goal_dict, action_map


def gen_task_param():
    n_subtasks = len(goal_set_labels)
    balance = [3]*6  # each context repeated 3 times; 3 distinct goal, mapping pairings, each repeated twice
    list_start_locations = []
    list_goals = []
    list_context = []
    list_action_map = []
    list_subtask = []

    contexts_a0 = randomize_context_order_with_autocorrelation(balance, repeat_probability=0.25, n_subtasks=n_subtasks)
    # contexts_a1 = randomize_context_order_with_autocorrelation(balance, repeat_probability=0.08)
    contexts_b = [(c + 6, subtask_idx) for c, reps in enumerate([3]*3) for subtask_idx in range(n_subtasks) for _ in range(reps)]
    np.random.shuffle(contexts_b)
    contexts = list(contexts_a0) + list(contexts_b)

    for ii, (ctx, subtask_idx) in enumerate(contexts):
        start_location, goal_dict, action_map = make_trial(ctx, subtask_idx)

        # define the trial and add to the lists
        list_start_locations.append(start_location)
        list_goals.append(goal_dict)
        list_context.append(ctx)
        list_subtask.append(subtask_idx)
        list_action_map.append(action_map)

    args = [list_start_locations, list_goals, list_context, list_subtask, list_action_map]
    kwargs = dict(grid_world_size=grid_world_size)
    return args, kwargs
