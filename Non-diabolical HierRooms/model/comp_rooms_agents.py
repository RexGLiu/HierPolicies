import numpy as np
import pandas as pd
from comp_rooms import GridWorld
from cython_library import GoalHypothesis, value_iteration
from cython_library import UpperDoorHypothesis
from cython_library import MappingHypothesis
from cython_library import HierarchicalHypothesis
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

    def update_goal_values(self, c, seq, goal, r):
        pass

    def prune_hypothesis_space(self, threshold=50.):
        pass

    def resample_hypothesis_space(self, n_particles):
        pass

    def augment_assignments(self, context):
        pass

    def count_hypotheses(self):
        return 1

    def navigate_rooms(self, evaluate=True, debug=False, prunning_threshold=None, comm=None):
        
        # initialize variables; index 0 corresponds to total in room, including sublvls
        step_counter = np.zeros((2*self.task.n_doors, self.task.n_trials))
        times_seen_room_ctx = np.zeros((2*self.task.n_doors, self.task.n_upper_ctx))
        steps_in_room_ctx = np.zeros((2*self.task.n_doors, self.task.n_upper_ctx))
        self.results = list()

        ii = 0
        min_particles = 100
        max_particles = 10000
        
        if comm is not None:
            rank = comm.Get_rank()
        else:
            rank = None

        if isinstance(self,HierarchicalAgent):
            clusterings = {
                'upper rooms' : [],
                'sublvl hierarchy' : [],
                'mappings hierarchy' : [],
                'door seq hierarchy' : [],
                'goals hierarchy' : [],
                'contexts' : []
                }
        else:
            clusterings = None

        
        while True:
            # get the current state and evaluate stop condition
            state = self.task.get_state()
            if state is None:
                break

            t = self.task.current_trial_number
            start_location, c, subroom = state
            subroom += 1
            c_idx = c/self.task.n_doors

            step_counter[0,t] += 1
            step_counter[subroom,t] += 1
            steps_in_room_ctx[0,c_idx] += 1
            steps_in_room_ctx[subroom,c_idx] += 1

            if step_counter[0,t] == 1:
                times_seen_room_ctx[0,c_idx] += 1

            if step_counter[subroom,t] == 1:
                times_seen_room_ctx[subroom,c_idx] += 1

            if steps_in_room_ctx[subroom,c_idx] == 1:
                self.resample_hypothesis_space(min_particles)
                self.augment_assignments(c)
                self.resample_hypothesis_space(max_particles)

            # select an action
            action = self.select_action(start_location)

            # take an action
            aa, end_location, goal_id, r = self.task.move(action)
            
            # update mapping
            self.update_mapping(c, action, aa)

            # End condition is a goal check
            if goal_id is not None:
                if subroom % 2 == 1:
                    seq = (subroom-1)/2
                else:
                    seq = None
                self.update_goal_values(c, seq, goal_id, r)

                if r == 1:
                    print rank, t, "Success", self.name, c, seq

                    if isinstance(self,HierarchicalAgent):
                        kk = np.argmax(self.log_belief)
                        h_MAP = self.hypotheses[kk]

                        clusterings['upper rooms'].append(h_MAP.get_upper_room_assignments())
                        clusterings['sublvl hierarchy'].append(h_MAP.get_complete_subroom_hierarchy())
                        clusterings['mappings hierarchy'].append(h_MAP.get_complete_mapping_hierarchy())
                        clusterings['door seq hierarchy'].append(h_MAP.get_complete_door_seq_hierarchy())
                        clusterings['goals hierarchy'].append(h_MAP.get_complete_goal_hierarchy())
                        clusterings['contexts'].append((c,seq))

            # stop criterion
            if step_counter[0,t] > 2000:
                print rank, t, "Stop", self.name
                return None


            trial_dict = {
                'Start Location': [start_location],
                'End Location': [end_location],
                'context': [c],
                'upper room context': [c_idx],
                'subroom': [subroom-1],
                'key-press': [action],
                'action': aa,  # the cardinal movement, in words
                'Reward Collected': [r],
                'Trial Number': [t],
                'In goal': not (self.task.current_trial_number == t),
                'Times Seen Context': times_seen_room_ctx[0,c_idx],
                'n actions taken in room': step_counter[0,t],
                'n actions taken in subroom': step_counter[subroom,t],
                'n actions taken in upper room': np.sum(step_counter[range(1,2*self.task.n_doors,2),t]),
                'n actions taken in sublvls': np.sum(step_counter[range(2,2*self.task.n_doors,2),t]),
                'Steps in Context': steps_in_room_ctx[0,c_idx],
                'Steps in Subcontext': steps_in_room_ctx[subroom,c_idx]
            }
            
            self.results.append(pd.DataFrame(trial_dict, index=[ii]))
            ii += 1

        return self.get_results(), clusterings


    def get_results(self):
        return pd.concat(self.results)


class FlatAgent(MultiStepAgent):

    def __init__(self, task, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.name = "Flat"

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        
        self.n_sublvls = self.task.get_n_sublvls()
        self.n_doors = self.task.get_n_doors()
        
        self.upper_door_hypotheses = UpperDoorHypothesis(self.n_doors, 1.0, goal_prior)
        self.mapping_hypotheses = MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              1.0, mapping_prior)
        self.subgoal_hypotheses = GoalHypothesis(self.n_doors, 1.0, goal_prior)


    def augment_assignments(self, context):
        # first figure out if context is in upper room or sublvl
        # then get corresponding hypotheses
        if context % self.n_doors == 0:
            # check if context is in start of upper level

            seq = self.task.get_door_order()
            h_g = self.upper_door_hypotheses
            assert type(h_g) is UpperDoorHypothesis
            h_g.add_new_context_assignment(context, context, seq)
        else:
            h_g = self.subgoal_hypotheses
            assert type(h_g) is GoalHypothesis
            h_g.add_new_context_assignment(context, context)
            
        h_m = self.mapping_hypotheses
        assert type(h_m) is MappingHypothesis
        h_m.add_new_context_assignment(context, context)

        # don't need to update the belief for the flat agent


    def update_goal_values(self, c, seq, goal, r):
        
        # figure out whether context is upper or sublvl and update accordingly
        if c % self.n_doors == 0:
            goal_idx_num = self.task.get_goal_index(0, goal)
            h_goal = self.upper_door_hypotheses
            assert type(h_goal) is UpperDoorHypothesis
            if h_goal.get_obs_likelihood(c, seq, goal_idx_num, r) > 1E-5:
                h_goal.update(c, seq, goal_idx_num, r)
            else:
                raise
        else:
            goal_idx_num = self.task.get_goal_index(1, goal)
            h_goal = self.subgoal_hypotheses
            assert type(h_goal) is GoalHypothesis
            if h_goal.get_obs_likelihood(c, goal_idx_num, r) > 1E-5:
                h_goal.update(c, goal_idx_num, r)
            else:
                raise


    def update_mapping(self, c, a, aa):
        h_m = self.mapping_hypotheses
        assert type(h_m) is MappingHypothesis

        if h_m.get_obs_likelihood(c, a, self.task.abstract_action_key[aa]) > 1E-4:
            h_m.update_mapping(c, a, self.task.abstract_action_key[aa])
        else:
            raise


    def get_goal_probability(self, context):
        # figure out if context is upper or sublvl
        if context % self.n_doors == 0:
            h_goal = self.upper_door_hypotheses
            assert type(h_goal) is UpperDoorHypothesis

            seq = self.task.get_door_order()
            goal_pmf = h_goal.get_goal_probability(context, seq)
        else:
            h_goal = self.subgoal_hypotheses
            assert type(h_goal) is GoalHypothesis

            goal_pmf = h_goal.get_goal_probability(context)

        return goal_pmf


    def get_mapping_function(self, context, aa):
        h_map = self.mapping_hypotheses
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
        
        h_map = self.mapping_hypotheses
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
        
        h_map = self.mapping_hypotheses
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
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.name = "Independent"
        
        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        self.alpha = alpha
        
        self.n_sublvls = self.task.get_n_sublvls()
        self.n_doors = self.task.get_n_doors()

        self.upper_door_hypotheses = [UpperDoorHypothesis(self.n_doors, self.alpha, goal_prior)]
        self.mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              self.alpha, mapping_prior)
        ]
        self.subgoal_hypotheses = [GoalHypothesis(self.n_doors, self.alpha, goal_prior)]
        
        # initialize the belief spaces for upper levels
        self.log_belief_door = np.ones(1, dtype=float)
        self.log_belief_mapping = np.ones(1, dtype=float)
        self.log_belief_subgoal = np.ones(1, dtype=float)


    def augment_assignments(self, context):
        _goal_hypotheses = list()
        _mapping_hypotheses = list()
        _goal_log_belief = list()
        _mapping_log_belief = list()


        if context % self.n_doors == 0:
            seq = self.task.get_door_order()

            for h_g in self.upper_door_hypotheses:
                assert type(h_g) is UpperDoorHypothesis

                old_assignments = h_g.get_assignments(seq)
                new_assignments = augment_assignments([old_assignments], context)

                for assignment in new_assignments:
                    k = assignment[context]
                    h_new = h_g.deep_copy()
                    h_new.add_new_context_assignment(context, k, seq)

                    _goal_hypotheses.append(h_new)
                    _goal_log_belief.append(h_new.get_log_posterior())

            self.upper_door_hypotheses = _goal_hypotheses
            self.log_belief_door = _goal_log_belief
        else:
            for h_g in self.subgoal_hypotheses:
                assert type(h_g) is GoalHypothesis

                old_assignments = h_g.get_assignments()
                new_assignments = augment_assignments([old_assignments], context)

                for assignment in new_assignments:
                    k = assignment[context]
                    h_new = h_g.deep_copy()
                    h_new.add_new_context_assignment(context, k)

                    _goal_hypotheses.append(h_new)
                    _goal_log_belief.append(h_new.get_log_posterior())

            self.subgoal_hypotheses = _goal_hypotheses
            self.log_belief_subgoal = _goal_log_belief
            
            
        # expand mapping hypothesis if in a new sublvl or first time in upper level
        if context % self.n_doors > 0 or self.task.get_door_order() == 0:
            for h_m in self.mapping_hypotheses:
                assert type(h_m) is MappingHypothesis

                old_assignments = h_m.get_assignments()
                new_assignments = augment_assignments([old_assignments], context)

                for assignment in new_assignments:
                    k = assignment[context]
                    h_new = h_m.deep_copy()
                    h_new.add_new_context_assignment(context, k)

                    _mapping_hypotheses.append(h_new)
                    _mapping_log_belief.append(h_new.get_log_posterior())

                self.mapping_hypotheses = _mapping_hypotheses
                self.log_belief_mapping = _mapping_log_belief
            

    def resample_hypothesis_space(self, n_particles):
        self.upper_door_hypotheses, self.log_belief_door = self.resample_particles(
                self.upper_door_hypotheses, self.log_belief_door, n_particles)
        
        self.mapping_hypotheses, self.log_belief_mapping = self.resample_particles(
                self.mapping_hypotheses, self.log_belief_mapping, n_particles)

        self.subgoal_hypotheses, self.log_belief_subgoal = self.resample_particles(
                self.subgoal_hypotheses, self.log_belief_subgoal, n_particles)

            
    def resample_particles(self, old_hypotheses, old_log_beliefs, n_particles):
        n_beliefs = len(old_log_beliefs)
        if n_beliefs > n_particles:
            old_hypotheses = np.array(old_hypotheses)
            old_log_beliefs = np.array(old_log_beliefs)
            
            # first select the n_max MAP hypotheses
            arg_idx = np.argpartition(old_log_beliefs, -n_particles)[-n_particles:]
            new_hypotheses = old_hypotheses[arg_idx]
            new_log_beliefs = old_log_beliefs[arg_idx]
            
            return new_hypotheses.tolist(), new_log_beliefs.tolist()
        else:
            return old_hypotheses, old_log_beliefs


    def update_goal_values(self, c, seq, goal, r):
        
        # figure out whether context is upper or sublvl and update accordingly
        if c % self.n_doors == 0:
            goal_idx_num = self.task.get_goal_index(0, goal)
            for h_goal in reversed(self.upper_door_hypotheses):
                assert type(h_goal) is UpperDoorHypothesis
                if h_goal.get_obs_likelihood(c, seq, goal_idx_num, r) > 1E-5:
                    h_goal.update(c, seq, goal_idx_num, r)
                else:
                    self.upper_door_hypotheses.remove(h_goal)

            # update the posterior of the goal hypotheses
            log_belief = np.zeros(len(self.upper_door_hypotheses))
            for ii, h_goal in enumerate(self.upper_door_hypotheses):
                assert type(h_goal) is UpperDoorHypothesis
                log_belief[ii] = h_goal.get_log_posterior()
            
            self.log_belief_door = log_belief
        else:
            goal_idx_num = self.task.get_goal_index(1, goal)
            for h_goal in reversed(self.subgoal_hypotheses):
                assert type(h_goal) is GoalHypothesis
                if h_goal.get_obs_likelihood(c, goal_idx_num, r) > 1E-5:
                    h_goal.update(c, goal_idx_num, r)
                else:
                    self.subgoal_hypotheses.remove(h_goal)

            # update the posterior of the goal hypotheses
            log_belief = np.zeros(len(self.subgoal_hypotheses))
            for ii, h_goal in enumerate(self.subgoal_hypotheses):
                assert type(h_goal) is GoalHypothesis
                log_belief[ii] = h_goal.get_log_posterior()

            self.log_belief_subgoal = log_belief


    def update_mapping(self, c, a, aa):
        for h_m in reversed(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            if h_m.get_obs_likelihood(c, a, self.task.abstract_action_key[aa]) > 1E-4:
                h_m.update_mapping(c, a, self.task.abstract_action_key[aa])
            else:
                self.mapping_hypotheses.remove(h_m)

        # update the posterior of the mapping hypothesis
        log_belief = np.zeros(len(self.mapping_hypotheses))
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            log_belief[ii] = h_m.get_log_posterior()
                
        self.log_belief_mapping = log_belief


    def get_goal_probability(self, context):
        # figure out if context is upper or sublvl
        if context % self.n_doors == 0:
            # get the value of the goals for the MAP cluster
            ii = np.argmax(self.log_belief_door)
            h_goal = self.upper_door_hypotheses[ii]

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
        ii = np.argmax(self.log_belief_mapping)
        h_map = self.mapping_hypotheses[ii]
        assert type(h_map) is MappingHypothesis

        mapping_pmf = np.zeros(self.task.n_primitive_actions, dtype=float)
        for a0 in range(self.task.n_primitive_actions):
            mapping_pmf[a0] = h_map.get_mapping_probability(context, a0, aa)

        return mapping_pmf
    
    def get_primitive_q(self, location):
        q_aa = self.get_abstract_action_q(location)
        c = self.task.get_current_context()

        # use the mapping distribution to get the q-values for the primitive actiosn
        q_a = np.zeros(self.task.n_primitive_actions)
        
        ii = np.argmax(self.log_belief_mapping)
        h_map = self.mapping_hypotheses[ii]
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
        
        ii = np.argmax(self.log_belief_mapping)
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



class HierarchicalAgent(IndependentClusterAgent):
    
    def __init__(self, task, alpha0=0.7, alpha1=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.name = "Hierarchical"

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
        self.n_sublvls = self.task.get_n_sublvls()
        self.n_doors = self.task.get_n_doors()
        self.n_primitive_actions = self.task.get_n_primitive_actions()
        self.n_abstract_actions = self.task.get_n_abstract_actions()
        self.goal_prior = goal_prior
        self.mapping_prior = mapping_prior
        
        # initialize the hypothesis space and beliefs for upper and sublvl doors/rewards
        self.hypotheses = [HierarchicalHypothesis(self.n_doors, self.n_primitive_actions, 
                                                              self.n_abstract_actions, self.inv_temp, 
                                                              self.gamma, self.stop_criterion, self.alpha0, 
                                                              self.alpha1, self.goal_prior, self.mapping_prior)]
        self.log_belief = np.ones(1, dtype=float)
        

    def augment_assignments(self, context):
        _new_hypotheses = list()
        _new_beliefs = list()
        del(self.log_belief)
        
        if context % self.n_doors == 0:
            seq = self.task.get_door_order()
            while self.hypotheses:
                h_parent = self.hypotheses.pop()
                h_children = h_parent.spawn_new_hypotheses(context, seq)
                _new_hypotheses += h_children
                del(h_parent)
        else:
            while self.hypotheses:
                h_parent = self.hypotheses.pop()
                h_children = h_parent.spawn_new_hypotheses(context)
                _new_hypotheses += h_children
                del(h_parent)
            
        self.hypotheses = _new_hypotheses
        for hypothesis in self.hypotheses:
            _new_beliefs.append(hypothesis.get_log_posterior())

        self.log_belief = np.array(_new_beliefs)


    def resample_hypothesis_space(self, n_particles):
        self.hypotheses, self.log_belief = self.resample_particles(
                self.hypotheses, self.log_belief, n_particles)
        
    def update_mapping(self, context, a, aa):
        for hypothesis in reversed(self.hypotheses): # reverse traversal needed to avoid unwanted Python side-effect
            if hypothesis.get_obs_mapping_likelihood(context, a, self.task.abstract_action_key[aa]) > 1E-4:
                hypothesis.updating_mapping(context, a, self.task.abstract_action_key[aa])
            else:
                self.hypotheses.remove(hypothesis)
                
        # update posteriors
        self.log_belief = np.zeros(len(self.hypotheses))
        for ii, h in enumerate(self.hypotheses):
            self.log_belief[ii] = h.get_log_posterior()

    def update_goal_values(self, context, seq, goal, r):
        if seq is None:
            seq = -1
            goal_idx_num = self.task.get_goal_index(1, goal)
        else:
            goal_idx_num = self.task.get_goal_index(0, goal)
        for hypothesis in reversed(self.hypotheses): # reverse traversal needed to avoid unwanted Python side-effect
            if hypothesis.get_obs_goal_likelihood(context, goal_idx_num, r, seq) > 1E-5:
                hypothesis.update(context, goal_idx_num, r, seq)
            else:
                self.hypotheses.remove(hypothesis)

        # update posteriors
        self.log_belief = np.zeros(len(self.hypotheses))
        for ii, h in enumerate(self.hypotheses):
            self.log_belief[ii] = h.get_log_posterior()

    def get_mapping_function(self, context, aa):
        ii = np.argmax(self.log_belief)
        h_MAP = self.hypotheses[ii]

        mapping_pmf = np.zeros(self.n_primitive_actions, dtype=float)
        for a0 in range(self.n_primitive_actions):
            mapping_pmf[a0] = h_MAP.get_mapping_probability(context, a0, aa)

        return mapping_pmf
            
    def get_action_pmf(self, location):
        c = self.task.get_current_context()
        q = self.get_abstract_action_q(location)

        # use softmax to convert to probability function
        p_aa = np.exp(self.inv_temp * q) / np.sum(np.exp(self.inv_temp * q))

        # use the distribution P(A=A*) to get P(a=a*) by integration
        # P(a=a*) = Sum[ P(a=A) x P(A=A*) ]
        pmf = np.zeros(self.n_primitive_actions)
        
        # get MAP hypothesis
        ii = np.argmax(self.log_belief)
        h_MAP = self.hypotheses[ii]
        
        for aa0 in np.arange(self.n_abstract_actions, dtype=np.int32):
            _mapping_pmf = np.zeros(self.n_primitive_actions)
            for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_MAP.get_mapping_probability(c, a0, aa0)
            pmf += _mapping_pmf * p_aa[aa0]

        # because we omit low probability goals from planning,
        # sometimes the pmf does not sum to one.
        # therefore, we need to re-normalize
        pmf /= pmf.sum()
        return pmf

    def get_goal_probability(self, context):
        # get MAP hypothesis
        ii = np.argmax(self.log_belief)
        h_MAP = self.hypotheses[ii]

        if context % self.n_doors == 0:
            seq = self.task.get_door_order()
            goal_pmf = h_MAP.get_goal_probability(context, seq)
        else:
            goal_pmf = h_MAP.get_goal_probability(context)

        return goal_pmf

    def get_primitive_q(self, location):
        q_aa = self.get_abstract_action_q(location)
        c = self.task.get_current_context()

        # use the mapping distribution to get the q-values for the primitive actiosn
        q_a = np.zeros(self.n_primitive_actions)
        
        # get MAP hypothesis
        ii = np.argmax(self.log_belief)
        h_MAP = self.hypotheses[ii]
            
        for aa0 in np.arange(self.n_abstract_actions, dtype=np.int32):
            _mapping_pmf = np.zeros(self.n_primitive_actions)
            for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_MAP.get_mapping_probability(c, a0, aa0)
            q_a += _mapping_pmf * q_aa[aa0]
            
        return q_a
