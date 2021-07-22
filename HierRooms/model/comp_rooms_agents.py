import numpy as np
import pandas as pd
from copy import deepcopy
from comp_rooms import GridWorld
from cython_library import GoalHypothesis, value_iteration
from cython_library import UpperDoorHypothesis
from cython_library import SublvlGoalHypothesis, SublvlMappingHypothesis
from cython_library import R_MappingHypothesis as MappingHypothesis
from cython_library import HierarchicalMappingHypothesis
from cython_library.assignments import *

""" these agents differ from the generative agents I typically use in that I need to pass a transition
function (and possibly a reward function) to the agent for each trial. """


def sample_cmf(cmf):
    return int(np.sum(np.random.rand() > cmf))


def make_q_primitive(q_abstract, mapping):
    q_primitive = np.zeros(8)
    n, m = np.shape(mapping)
    for aa in range(m):
        for a in range(n):
            q_primitive[a] += q_abstract[aa] * mapping[a, aa]
    return q_primitive


def kl_divergence(q, p):
    d = 0
    for q_ii, p_ii in zip(q, p):
        if p_ii > 0:
            d += p_ii * np.log2(p_ii/q_ii)
    return d


class MultiStepAgent(object):

    def __init__(self, task):
        self.task = task
        # assert type(self.task) is Task
        self.current_trial = 0
        self.results = []

    def get_action_pmf(self, location):
        assert type(location) is tuple
        return np.ones(self.task.n_primitive_actions, dtype=float) / self.task.n_primitive_actions

    def get_primitive_q(self, location):
        return np.ones(self.task.n_primitive_actions)

    def get_action_cmf(self, location):
        return np.cumsum(self.get_action_pmf(location))

    def get_goal_probability(self, context):
        return np.ones(self.task.n_goals, dtype=np.float) / self.task.n_goals

    def get_mapping_function(self, context, aa):
        return np.ones(self.task.n_primitive_actions, dtype=np.float) / self.task.n_primitive_actions

    def select_action(self, location):
        return sample_cmf(self.get_action_cmf(location))

    def update_mapping(self, c, a, aa):
        pass

    def update_goal_values(self, c, goal, r):
        pass

    def prune_hypothesis_space(self, threshold=50.):
        pass

    def resample_hypothesis_space(self):
        pass

    def augment_assignments(self, context):
        pass

    def count_hypotheses(self):
        return 1

    # def generate(self, evaluate=True, debug=False, prunning_threshold=None):

    #     # initialize variables
    #     step_counter = 0
    #     # results = list()
    #     times_seen_ctx = np.zeros(self.task.n_ctx)
    #     new_trial = True
    #     c = None
    #     t = None
    #     goal_locations = None
    #     kl_goal_pmf = None
    #     kl_map_pmf = None
    #     ii = 0

    #     while True:

    #         if new_trial:
    #             c = self.task.get_current_context()  # current context
    #             t = self.task.get_trial_number()  # current trial number
    #             goal_locations = self.task.get_goal_locations()
    #             new_trial = False
    #             times_seen_ctx[c] += 1
    #             step_counter = 0

    #             self.prune_hypothesis_space(threshold=prunning_threshold)
    #             if times_seen_ctx[c] == 1:
    #                 self.augment_assignments(c)

    #             if evaluate:
    #                 # compare the difference in goal probabilities
    #                 agent_goal_probability = self.get_goal_probability(c)
    #                 true_goal_probability = self.task.get_goal_values()
    #                 kl_goal_pmf = kl_divergence(agent_goal_probability, true_goal_probability)

    #         if evaluate:
    #             # compare the mapping probabilities for the greedy policy:
    #             kl_map_pmf = 0
    #             for aa0 in range(self.task.n_abstract_actions):
    #                 agent_mapping = self.get_mapping_function(c, aa0)
    #                 true_mapping = self.task.get_mapping_function(aa0)
    #                 kl_map_pmf += kl_divergence(agent_mapping, true_mapping)

    #         step_counter += 1

    #         # select an action
    #         start_location = self.task.get_location()
    #         action = self.select_action(start_location)

    #         # save for data output
    #         action_map = self.task.get_action_map()
    #         walls = self.task.get_walls()

    #         if debug:
    #             pmf = self.get_action_pmf(start_location)
    #             p = pmf[action]
    #         else:
    #             p = None

    #         # take an action
    #         aa, end_location, goal_id, r = self.task.move(action)

    #         # update mapping
    #         self.update_mapping(c, action, aa)

    #         # End condition is a goal check
    #         if goal_id is not None:
    #             self.update_goal_values(c, goal_id, r)
    #             new_trial = True

    #         results_dict = {
    #             'Context': c,
    #             'Start Location': [start_location],
    #             'Key-press': action,
    #             'End Location': [end_location],
    #             'Action Map': [action_map],
    #             'Walls': [walls],
    #             'Action': aa,  # the cardinal movement, in words
    #             'Reward': r,
    #             'In Goal': [goal_id is not None],
    #             'Chosen Goal': [goal_id],
    #             'Steps Taken': step_counter,
    #             'Goal Locations': [goal_locations],
    #             'Trial Number': [t],
    #             'Times Seen Context': times_seen_ctx[c],
    #             'Action Probability': [p],
    #         }
    #         if evaluate:
    #             results_dict['Goal KL Divergence'] = [kl_goal_pmf]
    #             results_dict['Map KL Divergence'] = [kl_map_pmf]
    #             results_dict['N Hypotheses'] = [self.count_hypotheses()]

    #         self.results.append(pd.DataFrame(results_dict, index=[ii]))
    #         ii += 1

    #         # evaluate stop condition
    #         if self.task.end_check():
    #             break

    #         if step_counter > 100:
    #             break

    #     return self.get_results()


    def navigate_rooms(self, evaluate=True, debug=False, prunning_threshold=None):
        
        # initialize variables
        step_counter = 0
        # results = list()
        times_seen_context = np.zeros(self.task.n_rooms*self.task.n_doors, dtype=np.uint64)
        new_trial = True
        r = None
        t = None
        goal_locations = None
        kl_goal_pmf = None
        kl_map_pmf = None
        ii = 0

        while True:

            if new_trial:
                # c = self.task.get_current_room()  # current room number
                c = self.task.get_current_context()  # current context
                t = self.task.get_trial_number()  # current trial number
                
                goal_locations = self.task.get_goal_locations()
                new_trial = False
                times_seen_context[c] += 1
                step_counter = 0
                
#                self.prune_hypothesis_space(threshold=prunning_threshold)
                self.resample_hypothesis_space()
                if times_seen_context[c] == 1:
                    self.augment_assignments(c)
                    
#                if evaluate:
#                    # compare the difference in goal probabilities
#                    agent_goal_probability = self.get_goal_probability(c)
#                    true_goal_probability = self.task.get_goal_values()
#                    kl_goal_pmf = kl_divergence(agent_goal_probability, true_goal_probability)
#
#            if evaluate:
#                # compare the mapping probabilities for the greedy policy:
#                kl_map_pmf = 0
#                for aa0 in range(self.task.n_abstract_actions):
#                    agent_mapping = self.get_mapping_function(c, aa0)
#                    true_mapping = self.task.get_mapping_function(aa0)
#                    kl_map_pmf += kl_divergence(agent_mapping, true_mapping)

            step_counter += 1

            # select an action
            start_location = self.task.get_location()
            action = self.select_action(start_location)
            
            # save for data output
            action_map = self.task.get_action_map()
            walls = self.task.get_walls()

            if debug:
                pmf = self.get_action_pmf(start_location)
                p = pmf[action]
            else:
                p = None

            # 
            if self.task.get_current_lvl() == 0:
                seq = self.task.get_door_order()
            else:
                seq = None
                
#            print c, ' ', times_seen_context[c], ' ', self.get_goal_probability(c)

            # take an action
            aa, end_location, goal_id, r = self.task.move(action)
            
            # update mapping
            self.update_mapping(c, action, aa)

            # End condition is a goal check
            if goal_id is not None:
#                print c, ' ', goal_id, ' ', r, ' ', times_seen_context[c]
                self.update_goal_values(c, seq, goal_id, r)
                new_trial = True
            

            results_dict = {
                'Room': c,
                'Start Location': [start_location],
                'Key-press': action,
                'End Location': [end_location],
                'Action Map': [action_map],
                'Walls': [walls],
                'Action': aa,  # the cardinal movement, in words
                'Reward': r,
                'In Goal': [goal_id is not None],
                'Chosen Goal': [goal_id],
                'Steps Taken': step_counter,
                'Goal Locations': [goal_locations],
                'Trial Number': [t],
                'Times Seen Context': times_seen_context[c],
                'Action Probability': [p],
                'Step': [1],
            }
#            if evaluate:
#                results_dict['Goal KL Divergence'] = [kl_goal_pmf]
#                results_dict['Map KL Divergence'] = [kl_map_pmf]
#                results_dict['N Hypotheses'] = [self.count_hypotheses()]

            self.results.append(pd.DataFrame(results_dict, index=[ii]))
            ii += 1

            # evaluate stop condition
            if self.task.end_check():
                print 'Successful completion'
                break

            if step_counter > 500:
                print 'Early termination'
                break

        return self.get_results()

    # def observe(self, subj_experience):
    #     """
    #     Take in the subject data, update the model at each step and return the posterior probability of the
    #     observations
    #     :param subj_experience: pd.DataFrame
    #     :return: vector of probabilities that the model would have selected the action
    #     """
    #
    #     # initialize variables
    #     results = [None] * len(subj_experience)
    #     columns = ['a%d' % a0 for a0 in range(self.task.n_primitive_actions)]
    #
    #     for ii, idx in enumerate(subj_experience.index):
    #
    #         # check that the updating works...
    #         t = subj_experience.loc[idx, 'Trial Number']
    #         if self.task.get_current_gridworld()_number != t:
    #             raise Exception('Task trial number does not match subject data')
    #
    #         # first, get the q-values over primitives
    #         start_location = subj_experience.loc[idx, 'Start Location']
    #         q = self.get_primitive_q(start_location)
    #         results[ii] = q
    #
    #         # update mapping (every step)
    #         c = subj_experience.loc[idx, 'Context']
    #         aa = subj_experience.loc[idx, 'Action']
    #         a = subj_experience.loc[idx, 'Key-press']
    #         self.update_mapping(c, a, aa)
    #
    #         # goal check, then update goal if in goal
    #         g = subj_experience.loc[idx, 'Chosen Goal']
    #         if g in self.task.goal_index.keys():
    #             r = int(subj_experience.loc[idx, 'Reward'])
    #             self.update_goal_values(c, g, r)
    #             self.task.start_next_trial()
    #
    #     return pd.DataFrame(data=results, index=subj_experience.index, columns=columns)

    def get_results(self):
        return pd.concat(self.results)


class FlatAgent(MultiStepAgent):

    def __init__(self, task, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        
        self.n_sublvls = self.task.get_n_sublvls()
        self.n_doors = self.task.get_n_doors()

        # initialize the hypothesis space for upper levels
        self.door_hypothesis = [UpperDoorHypothesis(self.n_doors, 1.0, goal_prior)]
        self.upper_mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              1.0, mapping_prior)
        ]
        
        # initialize the belief spaces for upper levels
        self.log_belief_door = np.ones(1, dtype=float)
        self.log_belief_upper_map = np.ones(1, dtype=float)
        
        # initialize the hypothesis space for sublevels
        self.subgoal_hypotheses = [GoalHypothesis(self.task.get_n_sublvl_doors(), 1.0, goal_prior)]
        self.submapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              1.0, mapping_prior)
            ]
        
        # initialize the belief spaces for sublevels
        self.log_belief_subgoal = np.ones(1, dtype=float)
        self.log_belief_submap = np.ones(1, dtype=float)


#    def count_hypotheses(self):
#        return len(self.log_belief_map)


    def augment_assignments(self, context):
        # first figure out if context is in upper room or sublvl
        # then get corresponding hypotheses
        if context % self.n_doors == 0:
            # check if context is in start of upper level
            h_g = self.door_hypothesis[0]
            h_m = self.upper_mapping_hypotheses[0]

            assert type(h_g) is UpperDoorHypothesis
            assert type(h_m) is MappingHypothesis
        else:
            h_g = self.subgoal_hypotheses[0]
            h_m = self.submapping_hypotheses[0]
        
            assert type(h_g) is GoalHypothesis
            assert type(h_m) is MappingHypothesis
            
        h_g.add_new_context_assignment(context, context)
        h_m.add_new_context_assignment(context, context)

        # don't need to update the belief for the flat agent


    def update_goal_values(self, c, seq, goal, r):
        
        # figure out whether context is upper or sublvl and update accordingly
        if c % self.n_doors == 0:
            goal_idx_num = self.task.get_goal_index(0, goal)
            for h_goal in self.door_hypothesis:
                assert type(h_goal) is UpperDoorHypothesis
                if h_goal.get_obs_likelihood(c, seq, goal_idx_num, r) > 0.0:
                    h_goal.update(c, seq, goal_idx_num, r)
                else:
                    self.door_hypothesis.remove(h_goal)

            # update the posterior of the goal hypotheses
            log_belief = np.zeros(len(self.door_hypothesis))
            for ii, h_goal in enumerate(self.door_hypothesis):
                assert type(h_goal) is UpperDoorHypothesis
                log_belief[ii] = h_goal.get_log_posterior()
            
            self.log_belief_door = log_belief
        else:
            goal_idx_num = self.task.get_goal_index(1, goal)
            for h_goal in self.subgoal_hypotheses:
                assert type(h_goal) is GoalHypothesis
                if h_goal.get_obs_likelihood(c, goal_idx_num, r) > 0.0:
                    h_goal.update(c, goal_idx_num, r)
                else:
                    self.door_hypothesis.remove(h_goal)

            # update the posterior of the goal hypotheses
            log_belief = np.zeros(len(self.subgoal_hypotheses))
            for ii, h_goal in enumerate(self.subgoal_hypotheses):
                assert type(h_goal) is GoalHypothesis
                log_belief[ii] = h_goal.get_log_posterior()

            self.log_belief_subgoal = log_belief


    def update_mapping(self, c, a, aa):
        if c % self.n_doors == 0:
            for h_m in self.upper_mapping_hypotheses:
                assert type(h_m) is MappingHypothesis
                if h_m.get_obs_likelihood(c, a, self.task.abstract_action_key[aa]) > 0.0:
                    h_m.update_mapping(c, a, self.task.abstract_action_key[aa])
                else:
                    self.door_hypothesis.remove(h_m)

            # update the posterior of the mapping hypothesis
            log_belief = np.zeros(len(self.upper_mapping_hypotheses))
            for ii, h_m in enumerate(self.upper_mapping_hypotheses):
                assert type(h_m) is MappingHypothesis
                log_belief[ii] = h_m.get_log_posterior()
                
            self.log_belief_upper_map = log_belief
        else:
            for h_m in self.submapping_hypotheses:
                assert type(h_m) is MappingHypothesis
                if h_m.get_obs_likelihood(c, a, self.task.abstract_action_key[aa]) > 0.0:
                    h_m.update_mapping(c, a, self.task.abstract_action_key[aa])
                else:
                    self.door_hypothesis.remove(h_m)

            # update the posterior of the mapping hypothesis
            log_belief = np.zeros(len(self.submapping_hypotheses))
            for ii, h_m in enumerate(self.submapping_hypotheses):
                assert type(h_m) is MappingHypothesis
                log_belief[ii] = h_m.get_log_posterior()
        
            self.log_belief_submap = log_belief


    def get_goal_probability(self, context):
        # figure out if context is upper or sublvl
        if context % self.n_doors == 0:
            # get the value of the goals for the MAP cluster
            ii = np.argmax(self.log_belief_door)
            h_goal = self.door_hypothesis[ii]

            assert type(h_goal) is UpperDoorHypothesis
            seq = self.task.get_door_order()
            goal_pmf = h_goal.get_goal_probability(context, seq)
        else:
            ii = np.argmax(self.log_belief_subgoal)
            h_goal = self.subgoal_hypotheses[ii]

            assert type(h_goal) is GoalHypothesis
            goal_pmf = h_goal.get_goal_probability(context)

        return goal_pmf


    def get_mapping_function(self, context, aa):
        if context % self.n_doors == 0:
            # used to calculate cross-entropy
            ii = np.argmax(self.log_belief_upper_map)
            h_map = self.upper_mapping_hypotheses[ii]
        else:
            ii = np.argmax(self.log_belief_submap)
            h_map = self.submapping_hypotheses[ii]

        assert type(h_map) is MappingHypothesis

        mapping_pmf = np.zeros(self.task.n_primitive_actions, dtype=float)
        for a0 in range(self.task.n_primitive_actions):
            mapping_pmf[a0] = h_map.get_mapping_probability(context, a0, aa)

        return mapping_pmf


    def convert_goal_values_to_reward(self, goal_pmf):
        grid = self.task.get_current_gridworld()
        assert type(grid) is GridWorld
        sublvl = self.task.get_current_lvl()

        reward_function = np.zeros(len(grid.state_location_key))
        for location, goal in grid.goal_locations.iteritems():
            goal_state = grid.state_location_key[location]
            p = goal_pmf[self.task.get_goal_index(sublvl, goal)]
            reward_function[goal_state] = p

        return reward_function


    def get_abstract_action_q(self, location): 
        c = self.task.get_current_context()
        t = self.task.get_transition_function()

        r = self.convert_goal_values_to_reward(self.get_goal_probability(c))
        v = value_iteration(t, r, self.gamma, self.stop_criterion)
        grid = self.task.get_current_gridworld()
        s = grid.state_location_key[location]

        # use the belman equation to get q-values
        q = np.zeros(self.task.n_abstract_actions)
        for aa in range(self.task.n_abstract_actions):
            q[aa] = np.sum(t[s, aa, :] * (r[:] + self.gamma * v[:]))

        return q


    def get_primitive_q(self, location):
        q_aa = self.get_abstract_action_q(location)
        c = self.task.get_current_context()

        # use the mapping distribution to get the q-values for the primitive actiosn
        q_a = np.zeros(self.task.n_primitive_actions)
        
        # get correct mapping hypothesis according to whether in upper or sublvl
        if c % self.n_doors == 0:
            ii = np.argmax(self.log_belief_upper_map)
            h_map = self.upper_mapping_hypotheses[ii]
        else:
            ii = np.argmax(self.log_belief_submap)
            h_map = self.submapping_hypotheses[ii]
            
        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):
            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
            q_a += _mapping_pmf * q_aa[aa0]
            
        return q_a


    def get_action_pmf(self, location):
        c = self.task.get_current_context()
        q = self.get_abstract_action_q(location)

        # use softmax to convert to probability function
        p_aa = np.exp(self.inv_temp * q) / np.sum(np.exp(self.inv_temp * q))

        # use the distribution P(A=A*) to get P(a=a*) by integration
        # P(a=a*) = Sum[ P(a=A) x P(A=A*) ]
        pmf = np.zeros(self.task.n_primitive_actions)
        
        # get correct mapping hypothesis according to whether in upper or sublvl
        if c % self.n_doors == 0:
            ii = np.argmax(self.log_belief_upper_map)
            h_map = self.upper_mapping_hypotheses[ii]
            
        else:
            ii = np.argmax(self.log_belief_submap)
            h_map = self.submapping_hypotheses[ii]
        
        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):
            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
            pmf += _mapping_pmf * p_aa[aa0]

        # because we omit low probability goals from planning,
        # sometimes the pmf does not sum to one.
        # therefore, we need to re-normalize
        pmf /= pmf.sum()
        return pmf


class IndependentClusterAgent(FlatAgent):

    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001, n_particles=1000, p_max = 0.1):
        super(FlatAgent, self).__init__(task)

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        self.alpha = alpha
        
        assert p_max <= 1 and p_max >=0
        self.n_particles = n_particles
        self.n_max = int(n_particles*p_max)
        
        self.n_sublvls = self.task.get_n_sublvls()
        self.n_doors = self.task.get_n_doors()


        # initialize the hypothesis space for upper levels with a single hypothesis that can be augmented
        # as new contexts are encountered
        self.door_hypothesis = [UpperDoorHypothesis(self.n_doors, alpha, goal_prior)]
        self.upper_mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              alpha, mapping_prior)
        ]
        
        # initialize the belief spaces for upper levels
        self.log_belief_door = np.ones(1, dtype=float)
        self.log_belief_upper_map = np.ones(1, dtype=float)
        
        # initialize the hypothesis space for sublevels
        self.subgoal_hypotheses = [SublvlGoalHypothesis(self.n_sublvls, 
                                                        self.task.get_n_sublvl_doors(), 
                                                        alpha, goal_prior)]
        self.submapping_hypotheses = [
            SublvlMappingHypothesis(self.n_sublvls, 
                                    self.task.n_primitive_actions, 
                                    self.task.n_abstract_actions,
                                    alpha, mapping_prior)
            ]
        
        # initialize the belief spaces for sublevels
        self.log_belief_subgoal = np.ones(1, dtype=float)
        self.log_belief_submap = np.ones(1, dtype=float)
        

#    def count_hypotheses(self):
#        return len(self.log_belief_map) + len(self.log_belief_goal)


    def augment_assignments(self, context):
        _goal_hypotheses = list()
        _mapping_hypotheses = list()
        _goal_log_belief = list()
        _mapping_log_belief = list()


        if context % self.n_doors == 0:
            for h_g in self.door_hypothesis:
                assert type(h_g) is UpperDoorHypothesis

                old_assignments = h_g.get_assignments()
                new_assignments = augment_assignments([old_assignments], context)

                # create a list of the new clusters to add
                for assignment in new_assignments:
                    k = assignment[context]
                    h_r0 = h_g.deep_copy()
                    h_r0.add_new_context_assignment(context, k)

                    _goal_hypotheses.append(h_r0)
                    _goal_log_belief.append(h_r0.get_log_posterior())

            for h_m in self.upper_mapping_hypotheses:
                assert type(h_m) is MappingHypothesis

                old_assignments = h_m.get_assignments()
                new_assignments = augment_assignments([old_assignments], context)

                for assignment in new_assignments:
                    k = assignment[context]
                    h_m0 = h_m.deep_copy()
                    h_m0.add_new_context_assignment(context, k)

                    _mapping_hypotheses.append(h_m0)
                    _mapping_log_belief.append(h_m0.get_log_posterior())

            self.upper_mapping_hypotheses = _mapping_hypotheses
            self.door_hypothesis = _goal_hypotheses
            self.log_belief_upper_map = _mapping_log_belief
            self.log_belief_door = _goal_log_belief
        else:
            for h_g in self.subgoal_hypotheses:
                assert type(h_g) is SublvlGoalHypothesis

                old_assignments = h_g.get_assignments()
                new_assignments = augment_sublvl_assignments([old_assignments], self.n_sublvls, context)

                # create a list of the new clusters to add
                for assignment in new_assignments:
                    sublvl = [ ii for ii, _subassignment in enumerate(assignment) if context in _subassignment.keys()]
                    assert len(sublvl) == 1
                    sublvl = sublvl[0]
                    k = assignment[sublvl][context]
                    h_r0 = h_g.deep_copy()
                    h_r0.add_new_context_assignment(sublvl, context, k)

                    _goal_hypotheses.append(h_r0)
                    _goal_log_belief.append(h_r0.get_log_posterior())

            for h_m in self.submapping_hypotheses:
                assert type(h_m) is SublvlMappingHypothesis

                old_assignments = h_m.get_assignments()
                new_assignments = augment_sublvl_assignments([old_assignments], self.n_sublvls, context)

                for assignment in new_assignments:
                    sublvl = [ ii for ii, _subassignment in enumerate(assignment) if context in _subassignment.keys()]
                    assert len(sublvl) == 1
                    sublvl = sublvl[0]
                    k = assignment[sublvl][context]
                    h_m0 = h_m.deep_copy()
                    h_m0.add_new_context_assignment(sublvl, context, k)

                    _mapping_hypotheses.append(h_m0)
                    _mapping_log_belief.append(h_m0.get_log_posterior())

            self.submapping_hypotheses = _mapping_hypotheses
            self.subgoal_hypotheses = _goal_hypotheses
            self.log_belief_submap = _mapping_log_belief
            self.log_belief_subgoal = _goal_log_belief
            

    def resample_hypothesis_space(self):
        self.door_hypothesis, self.log_belief_door = self.resample_particles(
                self.door_hypothesis, self.log_belief_door)
        
        self.upper_mapping_hypotheses, self.log_belief_upper_map = self.resample_particles(
                self.upper_mapping_hypotheses, self.log_belief_upper_map)

        self.subgoal_hypotheses, self.log_belief_subgoal = self.resample_particles(
                self.subgoal_hypotheses, self.log_belief_subgoal)
        
        self.submapping_hypotheses, self.log_belief_submap = self.resample_particles(
                self.submapping_hypotheses, self.log_belief_submap)

            
    def resample_particles(self, old_hypotheses, old_log_beliefs):
        n_beliefs = len(old_log_beliefs)
        if n_beliefs > self.n_particles:
            old_hypotheses = np.array(old_hypotheses)
            old_log_beliefs = np.array(old_log_beliefs)
            
            # first select the n_max MAP hypotheses
            arg_idx = np.argpartition(old_log_beliefs, -self.n_max)[-self.n_max:]
            new_hypotheses = old_hypotheses[arg_idx]
            new_log_beliefs = old_log_beliefs[arg_idx]
            
            # select out remaining elements
            arg_idx = np.isin(range(n_beliefs),arg_idx[-self.n_max:], invert=True)
            old_hypotheses = old_hypotheses[arg_idx]
            old_log_beliefs = old_log_beliefs[arg_idx]
            
            # randomly sample remaining elements according to their posterior
            p = np.exp(old_log_beliefs)
            p = p/np.sum(p)
            arg_idx = np.random.choice(n_beliefs-self.n_max, self.n_particles-self.n_max, replace=False)
            new_hypotheses = np.concatenate((new_hypotheses, old_hypotheses[arg_idx]))
            new_log_beliefs = np.concatenate((new_log_beliefs, old_log_beliefs[arg_idx]))
            
            return new_hypotheses.tolist(), new_log_beliefs.tolist()
        else:
            return old_hypotheses, old_log_beliefs

            
#    def prune_hypothesis_space(self, threshold=50.):
#        if threshold is not None:
#            _log_belief_door = []
#            _log_belief_upper_map = []
#            _door_hypothesis = []
#            _upper_mapping_hypotheses = []
#
#            _log_belief_subgoal = []
#            _log_belief_submap = []
#            _subgoal_hypotheses = []
#            _submapping_hypotheses = []
#
#            log_threshold = np.log(threshold)
#
#            # prune hypotheses for upper level
#            max_belief = np.max(self.log_belief_door)
#            for ii, log_b in enumerate(self.log_belief_door):
#                if max_belief - log_b < log_threshold:
#                    _log_belief_door.append(log_b)
#                    _door_hypothesis.append(self.door_hypothesis[ii])
#
#            max_belief = np.max(self.log_belief_upper_map)
#            for ii, log_b in enumerate(self.log_belief_upper_map):
#                if max_belief - log_b < log_threshold:
#                    _log_belief_upper_map.append(log_b)
#                    _upper_mapping_hypotheses.append(self.upper_mapping_hypotheses[ii])
#
#            self.log_belief_door = _log_belief_door
#            self.door_hypothesis = _door_hypothesis
#            self.log_belief_upper_map = _log_belief_upper_map
#            self.upper_mapping_hypotheses = _upper_mapping_hypotheses
#            
#            
#            # prune hypotheses for sublevels
#            max_belief = np.max(self.log_belief_subgoal)
#            for ii, log_b in enumerate(self.log_belief_subgoal):
#                if max_belief - log_b < log_threshold:
#                    _log_belief_subgoal.append(log_b)
#                    _subgoal_hypotheses.append(self.subgoal_hypotheses[ii])
#
#            max_belief = np.max(self.log_belief_submap)
#            for ii, log_b in enumerate(self.log_belief_submap):
#                if max_belief - log_b < log_threshold:
#                    _log_belief_submap.append(log_b)
#                    _submapping_hypotheses.append(self.submapping_hypotheses[ii])
#
#            self.log_belief_subgoal = _log_belief_subgoal
#            self.subgoal_hypotheses = _subgoal_hypotheses
#            self.log_belief_submap = _log_belief_submap
#            self.submapping_hypotheses = _submapping_hypotheses

            
    def update_goal_values(self, c, seq, goal, r):
        
        # figure out whether context is upper or sublvl and update accordingly
        if c % self.n_doors == 0:
            goal_idx_num = self.task.get_goal_index(0, goal)
            for h_goal in self.door_hypothesis:
                assert type(h_goal) is UpperDoorHypothesis
                if h_goal.get_obs_likelihood(c, seq, goal_idx_num, r) > 0.0:
                    h_goal.update(c, seq, goal_idx_num, r)
                else:
                    self.door_hypothesis.remove(h_goal)

            # update the posterior of the goal hypotheses
            log_belief = np.zeros(len(self.door_hypothesis))
            for ii, h_goal in enumerate(self.door_hypothesis):
                assert type(h_goal) is UpperDoorHypothesis
                log_belief[ii] = h_goal.get_log_posterior()
            
            self.log_belief_door = log_belief
        else:
            goal_idx_num = self.task.get_goal_index(1, goal)
            for h_goal in self.subgoal_hypotheses:
                assert type(h_goal) is SublvlGoalHypothesis
                if h_goal.get_obs_likelihood(c, goal_idx_num, r) > 0.0:
                    h_goal.update(c, goal_idx_num, r)
                else:
                    self.subgoal_hypotheses.remove(h_goal)

            # update the posterior of the goal hypotheses
            log_belief = np.zeros(len(self.subgoal_hypotheses))
            for ii, h_goal in enumerate(self.subgoal_hypotheses):
                assert type(h_goal) is SublvlGoalHypothesis
                log_belief[ii] = h_goal.get_log_posterior()

            self.log_belief_subgoal = log_belief


    def update_mapping(self, c, a, aa):
        if c % self.n_doors == 0:
            for h_m in self.upper_mapping_hypotheses:
                assert type(h_m) is MappingHypothesis
                if h_m.get_obs_likelihood(c, a, self.task.abstract_action_key[aa]) > 0.0:
                    h_m.update_mapping(c, a, self.task.abstract_action_key[aa])
                else:
                    self.upper_mapping_hypotheses.remove(h_m)


            # update the posterior of the mapping hypothesis
            log_belief = np.zeros(len(self.upper_mapping_hypotheses))
            for ii, h_m in enumerate(self.upper_mapping_hypotheses):
                assert type(h_m) is MappingHypothesis
                log_belief[ii] = h_m.get_log_posterior()
                
            self.log_belief_upper_map = log_belief
        else:
            for h_m in self.submapping_hypotheses:
                assert type(h_m) is SublvlMappingHypothesis
                if h_m.get_obs_likelihood(c, a, self.task.abstract_action_key[aa]) > 0.0:
                    h_m.update_mapping(c, a, self.task.abstract_action_key[aa])
                else:
                    self.submapping_hypotheses.remove(h_m)


            # update the posterior of the mapping hypothesis
            log_belief = np.zeros(len(self.submapping_hypotheses))
            for ii, h_m in enumerate(self.submapping_hypotheses):
                assert type(h_m) is SublvlMappingHypothesis
                log_belief[ii] = h_m.get_log_posterior()
        
            self.log_belief_submap = log_belief


    def get_goal_probability(self, context):
        # figure out if context is upper or sublvl
        if context % self.n_doors == 0:
            # get the value of the goals for the MAP cluster
            ii = np.argmax(self.log_belief_door)
            h_goal = self.door_hypothesis[ii]

            assert type(h_goal) is UpperDoorHypothesis
            seq = self.task.get_door_order()
            goal_pmf = h_goal.get_goal_probability(context, seq)
        else:
            ii = np.argmax(self.log_belief_subgoal)
            h_goal = self.subgoal_hypotheses[ii]

            assert type(h_goal) is SublvlGoalHypothesis
            goal_pmf = h_goal.get_goal_probability(context)

        return goal_pmf


    def get_mapping_function(self, context, aa):
        if context % self.n_doors == 0:
            # used to calculate cross-entropy
            ii = np.argmax(self.log_belief_upper_map)
            h_map = self.upper_mapping_hypotheses[ii]

            assert type(h_map) is MappingHypothesis
        else:
            ii = np.argmax(self.log_belief_submap)
            h_map = self.submapping_hypotheses[ii]
            
            assert type(h_map) is SublvlMappingHypothesis

        mapping_pmf = np.zeros(self.task.n_primitive_actions, dtype=float)
        for a0 in range(self.task.n_primitive_actions):
            mapping_pmf[a0] = h_map.get_mapping_probability(context, a0, aa)

        return mapping_pmf



class HierarchicalAgent(IndependentClusterAgent):
    
    def __init__(self, task, alpha0=1.0, alpha1=1.0, alpha2=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001, n_particles=1000, p_max = 0.1):
        super(FlatAgent, self).__init__(task)

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
        assert p_max <= 1 and p_max >=0
        self.n_particles = n_particles
        self.n_max = int(n_particles*p_max)
        
        self.n_sublvls = self.task.get_n_sublvls()
        self.n_doors = self.task.get_n_doors()
        self.mapping_prior = mapping_prior
        
        
        # initialize the hypothesis space and beliefs for upper and sublvl doors/rewards
        self.door_hypothesis = [UpperDoorHypothesis(self.n_doors, alpha0, goal_prior)]
        self.log_belief_door = np.ones(1, dtype=float)
        
        self.subgoal_hypotheses = [SublvlGoalHypothesis(self.n_sublvls, 
                                                        self.task.get_n_sublvl_doors(), 
                                                        alpha0, goal_prior)]
        self.log_belief_subgoal = np.ones(1, dtype=float)
        
        self.mapping_hypotheses = [HierarchicalMappingHypothesis(self.n_sublvls, 
                              self.task.n_primitive_actions, self.task.n_abstract_actions,
                              self.alpha0, self.alpha1, self.alpha2, self.mapping_prior)]
        self.mapping_log_belief = np.zeros(1, dtype=float)
        

#    def count_hypotheses(self):
#        return len(self.log_belief_map) + len(self.log_belief_goal)


    def augment_assignments(self, context):
        _goal_hypotheses = list()
        _mapping_hypotheses = list()
        _goal_log_belief = list()
        _mapping_log_belief = list()


        if context % self.n_doors == 0:
            for h_g in self.door_hypothesis:
                assert type(h_g) is UpperDoorHypothesis

                old_assignments = h_g.get_assignments()
                new_assignments = augment_assignments([old_assignments], context)

                # create a list of the new clusters to add
                for assignment in new_assignments:
                    k = assignment[context]
                    h_r0 = h_g.deep_copy()
                    h_r0.add_new_context_assignment(context, k)

                    _goal_hypotheses.append(h_r0)
                    _goal_log_belief.append(h_r0.get_log_posterior())

            for h_m in self.mapping_hypotheses:
                assert type(h_m) is HierarchicalMappingHypothesis

                old_assignments = h_m.get_room_assignments()
                new_assignments = augment_assignments([old_assignments], context)

                for assignment in new_assignments:
                    k = assignment[context]
                    h_m0 = h_m.deep_copy()
                    h_m0.add_new_room_context_assignment(context, k)

                    _mapping_hypotheses.append(h_m0)
                    _mapping_log_belief.append(h_m0.get_log_posterior())

            self.mapping_hypotheses = _mapping_hypotheses
            self.door_hypothesis = _goal_hypotheses
            self.mapping_log_belief = _mapping_log_belief
            self.log_belief_door = _goal_log_belief
        else:
            for h_g in self.subgoal_hypotheses:
                assert type(h_g) is SublvlGoalHypothesis

                old_assignments = h_g.get_assignments()
                new_assignments = augment_sublvl_assignments([old_assignments], self.n_sublvls, context)

                # create a list of the new clusters to add
                for assignment in new_assignments:
                    sublvl = [ ii for ii, _subassignment in enumerate(assignment) if context in _subassignment.keys()]
                    assert len(sublvl) == 1
                    sublvl = sublvl[0]
                    k = assignment[sublvl][context]
                    h_r0 = h_g.deep_copy()
                    h_r0.add_new_context_assignment(sublvl, context, k)

                    _goal_hypotheses.append(h_r0)
                    _goal_log_belief.append(h_r0.get_log_posterior())

            for h_m in self.mapping_hypotheses:
                assert type(h_m) is HierarchicalMappingHypothesis

                old_assignments = h_m.get_sublvl_assignments(context)
                new_assignments = augment_hierarchical_assignments([old_assignments], self.n_sublvls, context)

                for assignment in new_assignments:
                    h_m0 = h_m.deep_copy()
                    h_m0.add_new_sublvl_context_assignment(context, assignment)

                    _mapping_hypotheses.append(h_m0)
                    _mapping_log_belief.append(h_m0.get_log_posterior())

            self.mapping_hypotheses = _mapping_hypotheses
            self.subgoal_hypotheses = _goal_hypotheses
            self.mapping_log_belief = _mapping_log_belief
            self.log_belief_subgoal = _goal_log_belief


    def resample_hypothesis_space(self):
        self.door_hypothesis, self.log_belief_door = self.resample_particles(
                self.door_hypothesis, self.log_belief_door)
        
        self.subgoal_hypotheses, self.log_belief_subgoal = self.resample_particles(
                self.subgoal_hypotheses, self.log_belief_subgoal)
        
        self.mapping_hypotheses, self.mapping_log_belief = self.resample_particles(
                self.mapping_hypotheses, self.mapping_log_belief)


    def get_mapping_function(self, context, aa):
        ii = np.argmax(self.mapping_log_belief)
        h_map = self.mapping_hypotheses[ii]

        assert type(h_map) is HierarchicalMappingHypothesis

        mapping_pmf = np.zeros(self.task.n_primitive_actions, dtype=float)
        for a0 in range(self.task.n_primitive_actions):
            mapping_pmf[a0] = h_map.get_mapping_probability(context, a0, aa)

        return mapping_pmf
            

    def update_mapping(self, c, a, aa):
        for h_m in self.mapping_hypotheses:
            assert type(h_m) is HierarchicalMappingHypothesis
            if h_m.get_obs_likelihood(c, a, self.task.abstract_action_key[aa]) > 0.0:
                h_m.update_mapping(c, a, self.task.abstract_action_key[aa])

        # update the posterior of the mapping hypothesis
        log_belief = np.zeros(len(self.mapping_hypotheses))
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is HierarchicalMappingHypothesis
            log_belief[ii] = h_m.get_log_posterior()
                
        self.mapping_log_belief = log_belief


    def get_action_pmf(self, location):
        c = self.task.get_current_context()
        q = self.get_abstract_action_q(location)

        # use softmax to convert to probability function
        p_aa = np.exp(self.inv_temp * q) / np.sum(np.exp(self.inv_temp * q))

        # use the distribution P(A=A*) to get P(a=a*) by integration
        # P(a=a*) = Sum[ P(a=A) x P(A=A*) ]
        pmf = np.zeros(self.task.n_primitive_actions)
        
        # get MAP mapping hypothesis
        ii = np.argmax(self.mapping_log_belief)
        h_map = self.mapping_hypotheses[ii]
        
        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):
            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
            pmf += _mapping_pmf * p_aa[aa0]

        # because we omit low probability goals from planning,
        # sometimes the pmf does not sum to one.
        # therefore, we need to re-normalize
        pmf /= pmf.sum()
        return pmf
    


#class FlatMapPriorAgent(FlatAgent):
#
#    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
#                 mapping_prior=0.001, goal_prior=0.001):
#        super(FlatAgent, self).__init__(task)
#
#        self.gamma = gamma
#        self.inv_temp = inv_temp
#        self.stop_criterion = stop_criterion
#        self.alpha = alpha
#
#        # initialize the hypothesis space with a single hypothesis that can be augmented
#        # as new contexts are encountered
#        self.goal_hypotheses = [GoalHypothesis(self.task.n_goals, alpha, goal_prior)]
#        self.mapping_hypotheses = [
#            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
#                              alpha, mapping_prior)
#        ]
#
#        # initialize the belief spaces
#        self.log_belief_goal = np.ones(1, dtype=float)
#        self.log_belief_map = np.ones(1, dtype=float)
#
#    def count_hypotheses(self):
#        return len(self.log_belief_map) + len(self.log_belief_goal)
#
#    def augment_assignments(self, context):
#        _goal_hypotheses = list()
#        _mapping_hypotheses = list()
#        _goal_log_belief = list()
#        _mapping_log_belief = list()
#
#        for h_g in self.goal_hypotheses:
#            assert type(h_g) is GoalHypothesis
#
#            old_assignments = h_g.get_assignments()
#            new_assignments = augment_assignments([old_assignments], context)
#
#            # create a list of the new clusters to add
#            for assignment in new_assignments:
#                k = assignment[context]
#                h_r0 = h_g.deep_copy()
#                h_r0.add_new_context_assignment(context, k)
#
#                _goal_hypotheses.append(h_r0)
#                _goal_log_belief.append(h_r0.get_log_posterior())
#
#        for h_m in self.mapping_hypotheses:
#            assert type(h_m) is MappingHypothesis
#
#            old_assignments = h_m.get_assignments()
#            new_assignments = augment_assignments([old_assignments], context)
#
#            for assignment in new_assignments:
#                k = assignment[context]
#                h_m0 = h_m.deep_copy()
#                h_m0.add_new_context_assignment(context, k)
#
#                _mapping_hypotheses.append(h_m0)
#                _mapping_log_belief.append(h_m0.get_log_posterior())
#
#        self.mapping_hypotheses = _mapping_hypotheses
#        self.goal_hypotheses = _goal_hypotheses
#        self.log_belief_map = _mapping_log_belief
#        self.log_belief_goal = _goal_log_belief
#
#    def prune_hypothesis_space(self, threshold=50.):
#        if threshold is not None:
#            _log_belief_goal = []
#            _log_belief_map = []
#            _goal_hypotheses = []
#            _mapping_hypotheses = []
#
#            log_threshold = np.log(threshold)
#
#            max_belief = np.max(self.log_belief_goal)
#            for ii, log_b in enumerate(self.log_belief_goal):
#                if max_belief - log_b < log_threshold:
#                    _log_belief_goal.append(log_b)
#                    _goal_hypotheses.append(self.goal_hypotheses[ii])
#
#            max_belief = np.max(self.log_belief_map)
#            for ii, log_b in enumerate(self.log_belief_map):
#                if max_belief - log_b < log_threshold:
#                    _log_belief_map.append(log_b)
#                    _mapping_hypotheses.append(self.mapping_hypotheses[ii])
#
#            self.log_belief_goal = _log_belief_goal
#            self.goal_hypotheses = _goal_hypotheses
#            self.log_belief_map = _log_belief_map
#            self.mapping_hypotheses = _mapping_hypotheses
#
#    def update_mapping(self, c, a, aa):
#        for h_m in self.mapping_hypotheses:
#            assert type(h_m) is MappingHypothesis
#            h_m.update_mapping(c, a, self.task.abstract_action_key[aa])
#
#        # update the posterior of the mapping hypothesis
#        log_belief = np.zeros(len(self.mapping_hypotheses))
#        for ii, h_m in enumerate(self.mapping_hypotheses):
#            assert type(h_m) is MappingHypothesis
#            log_belief[ii] = h_m.get_log_likelihood()
#
#        self.log_belief_map = log_belief
#
#    def get_goal_prior_over_new_contexts(self):
#        from cython_library.core import get_prior_log_probability
#        log_prior_pmf = []
#        goal_pmfs = []
#
#        for h_goal in self.goal_hypotheses:
#
#            set_assignment = h_goal.get_set_assignments()
#            n_k = np.max(set_assignment) + 1
#            ll = h_goal.get_log_likelihood()
#
#            for ts in range(n_k):
#                sa0 = np.array(np.concatenate([set_assignment, [ts]]), dtype=np.int32)
#                log_prior_pmf.append(get_prior_log_probability(sa0, self.alpha) + ll)
#                goal_pmfs.append(h_goal.get_set_goal_probability(ts))
#
#            # new cluster
#            sa0 = np.array(np.concatenate([set_assignment, [n_k]]), dtype=np.int32)
#            log_prior_pmf.append(get_prior_log_probability(sa0, self.alpha) + ll)
#            goal_pmfs.append(np.ones(self.task.n_goals, dtype=np.float32) / self.task.n_goals)
#
#        # Normalize the prior
#        log_prior_pmf = np.array(log_prior_pmf)
#        log_prior_pmf -= np.max(log_prior_pmf)
#        prior_pmf = np.exp(log_prior_pmf)
#        prior_pmf /= np.sum(prior_pmf)
#
#        # weight the goal probability to create a distribution over goals
#        goal_pmf = np.squeeze(np.dot(np.array(goal_pmfs).T, np.array([prior_pmf]).T))
#
#        goal_pmf = pd.DataFrame({
#            'Probability': goal_pmf,
#            'Goal': range(1, self.task.n_goals + 1),
#            'Map': ['Combined'] * self.task.n_goals,
#            'Model': ['NoMapPrior'] * self.task.n_goals
#        })
#        return goal_pmf
#
#
#class JointClusteringAgent(MultiStepAgent):
#
#    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
#                 mapping_prior=0.001, goal_prior=0.001):
#        super(JointClusteringAgent, self).__init__(task)
#
#        self.gamma = gamma
#        self.inv_temp = inv_temp
#        self.stop_criterion = stop_criterion
#        self.alpha = alpha
#
#        # initialize the hypothesis space with a single hypothesis that can be augmented
#        # as new contexts are encountered
#        self.goal_hypotheses = [GoalHypothesis(self.task.n_goals, alpha, goal_prior)]
#        self.mapping_hypotheses = [
#            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
#                              alpha, mapping_prior)
#        ]
#
#        # initialize the belief spaces
#        self.log_belief = np.zeros(1, dtype=float)
#        self.map_likelihood = np.zeros(1, dtype=float)
#        self.goal_likelihood = np.zeros(1, dtype=float)
#
#    def augment_assignments(self, context):
#        _goal_hypotheses = list()
#        _mapping_hypotheses = list()
#
#        for h_g, h_m in zip(self.goal_hypotheses, self.mapping_hypotheses):
#            assert type(h_g) is GoalHypothesis
#            assert type(h_m) is MappingHypothesis
#
#            old_assignments = h_g.get_assignments()
#            new_assignments = augment_assignments([old_assignments], context)
#
#            for assignment in new_assignments:
#                k = assignment[context]
#                h_g0 = h_g.deep_copy()
#                h_g0.add_new_context_assignment(context, k)
#
#                h_m0 = h_m.deep_copy()
#                h_m0.add_new_context_assignment(context, k)
#
#                _goal_hypotheses.append(h_g0)
#                _mapping_hypotheses.append(h_m0)
#
#        self.mapping_hypotheses = _mapping_hypotheses
#        self.goal_hypotheses = _goal_hypotheses
#
#        self.update_goal_log_likelihood()
#        self.update_mapping_loglikelihood()
#        self.update_belief()
#
#    def update_mapping_loglikelihood(self):
#        self.map_likelihood = np.zeros(len(self.mapping_hypotheses))
#        for ii, h_map in enumerate(self.mapping_hypotheses):
#            self.map_likelihood[ii] = h_map.get_log_likelihood()
#
#    def update_goal_log_likelihood(self):
#        self.goal_likelihood = np.zeros(len(self.goal_hypotheses))
#        for ii, h_goal in enumerate(self.goal_hypotheses):
#            self.goal_likelihood[ii] = h_goal.get_log_likelihood()
#
#    def update_belief(self):
#        log_posterior = np.zeros(len(self.mapping_hypotheses))
#        for ii, h_map in enumerate(self.mapping_hypotheses):
#            log_posterior[ii] = h_map.get_log_prior() + self.map_likelihood[ii] + \
#                                self.goal_likelihood[ii]
#        self.log_belief = log_posterior
#
#    def count_hypotheses(self):
#        return len(self.log_belief)
#
#    def update_goal_values(self, c, goal, r):
#        goal_idx_num = self.task.get_goal_index(goal)
#        for h_goal in self.goal_hypotheses:
#            assert type(h_goal) is GoalHypothesis
#            h_goal.update(c, goal_idx_num, r)
#
#        # update the belief distribution
#        self.update_goal_log_likelihood()
#        self.update_belief()
#
#    def update_mapping(self, c, a, aa):
#        for h_map in self.mapping_hypotheses:
#            assert type(h_map) is MappingHypothesis
#            h_map.update_mapping(c, a, self.task.abstract_action_key[aa])
#
#        # update the belief distribution
#        self.update_mapping_loglikelihood()
#        self.update_belief()
#
#    def prune_hypothesis_space(self, threshold=50.):
#        if threshold is not None:
#            _goal_hypotheses = []
#            _mapping_hypotheses = []
#            _goal_log_likelihoods = []
#            _mapping_log_likelihoods = []
#            _log_belief = []
#
#            max_belief = np.max(self.log_belief)
#            log_threshold = np.log(threshold)
#
#            for ii, log_b in enumerate(self.log_belief):
#                if max_belief - log_b < log_threshold:
#                    _log_belief.append(log_b)
#                    _goal_log_likelihoods.append(self.goal_likelihood[ii])
#                    _mapping_log_likelihoods.append(self.map_likelihood[ii])
#
#                    _goal_hypotheses.append(self.goal_hypotheses[ii])
#                    _mapping_hypotheses.append(self.mapping_hypotheses[ii])
#
#            self.goal_hypotheses = _goal_hypotheses
#            self.mapping_hypotheses = _mapping_hypotheses
#
#            self.goal_likelihood = _goal_log_likelihoods
#            self.map_likelihood = _mapping_log_likelihoods
#            self.log_belief = _log_belief
#
#    def get_goal_probability(self, context):
#
#        # get the value of the goals of the MAP cluster
#        ii = np.argmax(self.log_belief)
#        h_goal = self.goal_hypotheses[ii]
#        assert type(h_goal) is GoalHypothesis
#
#        goal_expectation = h_goal.get_goal_probability(context)
#
#        return goal_expectation
#
#    def get_mapping_function(self, context, aa):
#        # used to calculate cross entropy
#        ii = np.argmax(self.log_belief)
#
#        h_map = self.mapping_hypotheses[ii]
#        assert type(h_map) is MappingHypothesis
#
#        mapping_expectation = np.zeros(self.task.n_primitive_actions, dtype=float)
#        for a0 in range(self.task.n_primitive_actions):
#            mapping_expectation[a0] += h_map.get_mapping_probability(context, a0, aa)
#
#        return mapping_expectation
#
#    def convert_goal_values_to_reward(self, goal_pmf):
#        grid = self.task.get_current_gridworld()
#        assert type(grid) is GridWorld
#
#        reward_function = np.zeros(len(grid.state_location_key))
#        for location, goal in grid.goal_locations.iteritems():
#            goal_state = grid.state_location_key[location]
#            p = goal_pmf[self.task.get_goal_index(goal)]
#            reward_function[goal_state] = p - 0.1 * (1 - p)
#
#        return reward_function
#
#    def get_abstract_action_q(self, location):
#        c = self.task.get_current_context()
#        t = self.task.get_transition_function()
#
#        r = self.convert_goal_values_to_reward(self.get_goal_probability(c))
#        v = value_iteration(t, r, self.gamma, self.stop_criterion)
#        s = self.task.state_location_key[location]
#
#        # use the belman equation to get q-values
#        q = np.zeros(self.task.n_abstract_actions)
#        for aa in range(self.task.n_abstract_actions):
#            q[aa] = np.sum(t[s, aa, :] * (r[:] + self.gamma * v[:]))
#
#        return q
#
#    def get_primitive_q(self, location):
#        q_aa = self.get_abstract_action_q(location)
#        c = self.task.get_current_context()
#
#        # use the mapping distribution to get the q-values for the primitive actiosn
#        q_a = np.zeros(self.task.n_primitive_actions)
#        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):
#            ii = np.argmax(self.log_belief)
#            h_map = self.mapping_hypotheses[ii]
#
#            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
#            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
#                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
#            q_a += _mapping_pmf * q_aa[aa0]
#
#        return q_a
#
#    def get_action_pmf(self, location, threshold=0.01):
#
#        c = self.task.get_current_context()
#        q = self.get_abstract_action_q(location)
#
#        # use softmax to convert to probability function
#        p_aa = np.exp(self.inv_temp * q) / np.sum(np.exp(self.inv_temp * q))
#
#        # use the distribution P(A=A*) to get P(a=a*) by integration
#        # P(a=a*) = Sum[ P(a=A) x P(A=A*) ]
#        pmf = np.zeros(self.task.n_primitive_actions)
#        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):
#
#            # get a mapping probability
#            # for each abstract action
#            ii = np.argmax(self.log_belief)
#            h_map = self.mapping_hypotheses[ii]
#
#            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
#            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
#                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
#            pmf += _mapping_pmf * p_aa[aa0]
#
#        # because we omit low probability goals from planning, sometimes the pmf does not sum to one.
#        # therefore, we need to re-normalize
#        pmf /= pmf.sum()
#        return pmf
#
#    def get_goal_prior_over_new_contexts(self):
#        from cython_library.core import get_prior_log_probability
#
#        all_log_prior_pmfs = [list() for _ in range(len(self.task.list_action_maps))]
#        goal_pmfs = []
#
#        aa_key = self.task.abstract_action_key
#
#        for h_goal, h_map in zip(self.goal_hypotheses, self.mapping_hypotheses):
#
#            set_assignment = h_map.get_set_assignments()
#            n_k = np.max(set_assignment) + 1
#
#            for ts in range(n_k):
#                sa0 = np.array(np.concatenate([set_assignment, [ts]]), dtype=np.int32)
#                goal_pmfs.append(h_goal.get_set_goal_probability(ts))
#
#                for m_idx, action_map in enumerate(self.task.list_action_maps):
#                    ll = h_map.get_log_likelihood() + h_goal.get_log_likelihood()
#                    for key, dir_ in action_map.iteritems():
#                        ll += np.log(h_map.get_pr_aa_given_a_ts(ts, key, aa_key[dir_]))
#                    all_log_prior_pmfs[m_idx].append(ll + get_prior_log_probability(sa0, self.alpha))
#
#            # new cluster
#            goal_pmfs.append(np.ones(self.task.n_goals, dtype=np.float32) / self.task.n_goals)
#            for m_idx, action_map in enumerate(self.task.list_action_maps):
#                sa0 = np.array(np.concatenate([set_assignment, [n_k]]), dtype=np.int32)
#                ll = h_map.get_log_likelihood() + h_goal.get_log_likelihood()
#                ll += np.log(self.task.n_abstract_actions / float(self.task.n_primitive_actions))
#                all_log_prior_pmfs[m_idx].append(ll + get_prior_log_probability(sa0, self.alpha))
#
#        # Normalize the prior
#        def normalize_prior(log_prior_pmf):
#            log_prior_pmf = np.array(log_prior_pmf)
#            log_prior_pmf -= np.max(log_prior_pmf)
#            prior_pmf = np.exp(log_prior_pmf)
#            prior_pmf /= np.sum(prior_pmf)
#            return prior_pmf
#
#        results = []
#        for m_idx, action_map in enumerate(self.task.list_action_maps):
#            prior_pmf = normalize_prior(all_log_prior_pmfs[m_idx])
#
#            # weight the goal probability to create a distribution over goals
#            goal_pmf = np.squeeze(np.dot(np.array(goal_pmfs).T, np.array([prior_pmf]).T))
#            results.append(pd.DataFrame({
#                'Probability': goal_pmf,
#                'Goal': range(1, self.task.n_goals + 1),
#                'Map': [m_idx] * self.task.n_goals,
#                'Model': ['TS'] * self.task.n_goals,
#                'Action Map': [action_map] * self.task.n_goals,
#            }))
#        return pd.concat(results)


