import numpy as np
import pandas as pd

from gridworld import Task
from cython_library import RewardHypothesis, MappingHypothesis, HierarchicalHypothesis
from cython_library import policy_iteration
from cython_library.assignments import *
from scipy.misc import logsumexp

""" these agents differ from the generative agents I typically use in that I need to pass a transition
function (and possibly a reward function) to the agent for each trial. """


def make_q_primitive(q_abstract, mapping):
    q_primitive = np.zeros(8)
    n, m = np.shape(mapping)
    for aa in range(m):
        for a in range(n):
            q_primitive[a] += q_abstract[aa] * mapping[a, aa]
    return q_primitive

def softmax_to_pdf(q_values, inverse_temperature):
    pdf = np.exp(np.array(q_values) * float(inverse_temperature))
    pdf = pdf / np.sum(pdf)
    return pdf


def sample_cmf(cmf):
    return int(np.sum(np.random.rand() > cmf))




class MultiStepAgent(object):

    def __init__(self, task, n_particles=300, p_max = 1., sample_tau = 1):
        self.task = task
        assert type(self.task) is Task
        self.current_trial = 0

        # p_max specifies what proportion of n_particles are to be the highest probability particles
        # remainder are sampled randomly from the posterior
        assert p_max <= 1 and p_max >=0
        self.n_particles = n_particles
        self.n_max = int(n_particles*p_max)
        self.sample_tau = sample_tau

    def get_action_pmf(self, state):
        return np.ones(self.task.n_primitive_actions, dtype=float) / self.task.n_primitive_actions

    def get_action_cmf(self, state):
        return np.cumsum(self.get_action_pmf(state))

    def select_action(self, state):
        return sample_cmf(self.get_action_cmf(state))

    def update(self, experience_tuple):
        pass

    def get_reward_function(self, state):
        pass

    def resample_hypothesis_space(self, n_particles=None):
        pass

    def resample_hypotheses(self, old_hypotheses, old_log_beliefs, n_particles=None):
        if n_particles is None:
            n_particles = self.n_particles
        
        n_beliefs = len(old_log_beliefs)
        if n_beliefs > n_particles:
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
            p = old_log_beliefs - np.amax(old_log_beliefs)  # shift logits so that exp does not yield tiny values
            p = np.exp(p)
            p = p/np.sum(p)
            idx_set = np.arange(n_beliefs-self.n_max)
            assert p.size == idx_set.size
            idx_set = idx_set[p>0]
            p = p[p>0]

            # check if number of particles with non-zero probability left still exceeds n_particles
            if p.size > n_particles-self.n_max:
                arg_idx = np.random.choice(idx_set, n_particles-self.n_max, p=p, replace=False)
            else:
                arg_idx = idx_set

            new_hypotheses = np.concatenate((new_hypotheses, old_hypotheses[arg_idx]))
            new_log_beliefs = np.concatenate((new_log_beliefs, old_log_beliefs[arg_idx]))
            
            return new_log_beliefs, new_hypotheses
        
        else:
            return old_log_beliefs, old_hypotheses

    def augment_assignments(self, context):
        pass

    def evaluate_mixing_agent(self, xp, yp, c, r):
        pass

    def is_meta(self):
        return False

    def get_joint_probability(self):
        pass
    
    def generate(self, pruning_threshold=1000):
        """ run through all of the trials of a task and output trial-by-trial data"""

        # count the number of steps to completion
        step_counter = np.zeros(self.task.n_trials)
        results = list()
        times_seen_ctx = np.zeros(self.task.n_ctx)
        steps_in_ctx = np.zeros(self.task.n_ctx)

        if isinstance(self,HierarchicalAgent):
            list_clusterings = []
        else:
            list_clusterings = None
            
        t = -1
        ii = 0
        while True:

            # get the current state and evaluate stop condition
            state = self.task.get_state()
            if state is None:
                break
            
            t_prev = t
            t = self.task.current_trial_number
            step_counter[t] += 1

            _, c = state
            steps_in_ctx[c] += 1

            if step_counter[t] == 1:
                times_seen_ctx[c] += 1

                # self.resample_hypothesis_space()
                # entering a new context, augment for new context
                if times_seen_ctx[c] == 1:
                    # augment the clustering assignments
                    self.resample_hypothesis_space()
                    self.augment_assignments(c)
                    self.resample_hypothesis_space(10000)

            if (t_prev != t) and isinstance(self,HierarchicalAgent):
                kk = np.argmax(self.log_belief)
                h_room = self.rooms_hypotheses[kk]
                
                room_assgn = h_room.get_room_assignments()
                reward_assgn = h_room.get_all_reward_assignments()
                mapping_assgn = h_room.get_all_mapping_assignments()
                new_state = state[0]
                
                rew_fn = h_room.get_reward_function(c)


                clusterings = {'start state': state,
                       'rooms' : [room_assgn],
                       'upper rewards' : [reward_assgn[0]],
                       'upper mappings' : [mapping_assgn[0]],
                       'lower rewards' : [reward_assgn[1]],
                       'lower mappings' : [mapping_assgn[1]],
                       'contexts' : [c],
                       'state' : [new_state],
                       'rew fn' : [rew_fn]
                       }
                
            # select an action
            action = self.select_action(state)

            # save for data output
            action_map = self.task.current_trial.action_map
            goal_location = self.task.current_trial.goal_location
            walls = self.task.current_trial.walls
            inverse_abstract_action_key = self.task.current_trial.inverse_abstract_action_key

            # take an action
            experience_tuple = self.task.move(action)
            ((x, y), c), a, aa, r, ((xp, yp), _) = experience_tuple

            self.evaluate_mixing_agent(xp, yp, c, r)

            # update the learner
            self.update(experience_tuple)

            trial_dict = {
                'Start Location': [(x, y)],
                'End Location': [(xp, yp)],
                'context': [c],
                'key-press': [action],
                'action': [inverse_abstract_action_key[aa]],  # the cardinal movement, in words
                'Reward Collected': [r],
                'n actions taken': step_counter[t],
                'Trial Number': [t],
                'In goal': not (self.task.current_trial_number == t),
                'Times Seen Context': times_seen_ctx[c],
                'action_map': [action_map],
                'goal location': [goal_location],
                'walls': [walls],
                'Steps in Context': steps_in_ctx[c]
            }

            results.append(pd.DataFrame(trial_dict, index=[ii]))
            
            if step_counter[t] > 500:
                print('Early termination', c, step_counter[t])

                trial_dict['Success'] = False
                results.append(pd.DataFrame(trial_dict, index=[ii]))
                ii += 1
                break

            # store clusterings if in goal
            if isinstance(self,HierarchicalAgent):
                kk = np.argmax(self.log_belief)
                h_room = self.rooms_hypotheses[kk]
                
                prev_room_assgn = room_assgn
                prev_reward_assgn = reward_assgn
                prev_mapping_assgn = mapping_assgn
                prev_state = new_state
                prev_rew_fn = rew_fn
                
                room_assgn = h_room.get_room_assignments()
                reward_assgn = h_room.get_all_reward_assignments()
                mapping_assgn = h_room.get_all_mapping_assignments()
                new_state = (xp, yp)
                
                rew_fn = h_room.get_reward_function(c)
                
                if ((xp, yp) in self.task.list_goal_locations):
                    
                    # store clustering just before agent receives first goal feedback
                    if len(clusterings['contexts']) == 1:
                        clusterings['rooms'].append(prev_room_assgn)
                        clusterings['upper rewards'].append(prev_reward_assgn[0])
                        clusterings['upper mappings'].append(prev_mapping_assgn[0])
                        clusterings['lower rewards'].append(prev_reward_assgn[1])
                        clusterings['lower mappings'].append(prev_mapping_assgn[1])
                        clusterings['contexts'].append(c)
                        clusterings['state'].append(prev_state)
                        clusterings['rew fn'].append(prev_rew_fn)
                    
                    
                    clusterings['rooms'].append(room_assgn)
                    clusterings['upper rewards'].append(reward_assgn[0])
                    clusterings['upper mappings'].append(mapping_assgn[0])
                    clusterings['lower rewards'].append(reward_assgn[1])
                    clusterings['lower mappings'].append(mapping_assgn[1])
                    clusterings['contexts'].append(c)
                    clusterings['state'].append(new_state)
                    clusterings['rew fn'].append(rew_fn)
                
                if not (self.task.current_trial_number == t):
                    list_clusterings.append(clusterings)
                    
            ii += 1

        return pd.concat(results), list_clusterings


class FullInformationAgent(MultiStepAgent):
    """ this agent uses the reward function and transition function to solve the task exactly.
    """

    def __init__(self, task, discount_rate=0.8, iteration_criterion=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions

    def select_abstract_action(self, state):
        (x, y), c = state

        # what is current state?
        s = self.task.state_location_key[(x, y)]

        pi = policy_iteration(self.task.current_trial.transition_function,
                              self.task.current_trial.reward_function[s, :],
                              gamma=self.gamma,
                              stop_criterion=self.iteration_criterion)

        # use the policy to choose the correct action for the current state
        abstract_action = pi[s]

        return abstract_action

    def select_action(self, state):
        (x, y), c = state

        abstract_action = self.select_abstract_action(state)

        # use the actual action_mapping to get the correct primitive action key
        inverse_abstract_action_key = {aa: move for move, aa in self.task.current_trial.abstract_action_key.iteritems()}
        inverse_action_map = {move: key_press for key_press, move in self.task.current_trial.action_map.iteritems()}

        move = inverse_abstract_action_key[abstract_action]
        key_press = inverse_action_map[move]
        return key_press


class ModelBasedAgent(FullInformationAgent):
    """ This Agent learns the reward function and mapping will model based planning
    """

    def __init__(self, task, discount_rate=0.8, iteration_criterion=0.01, mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

        # mappings!
        self.mapping_history = np.ones((self.task.n_ctx, self.task.n_primitive_actions, self.task.n_abstract_actions+1),
                                        dtype=float) * mapping_prior
        self.abstract_action_counts = np.ones((self.task.n_ctx, self.task.n_abstract_actions+1), dtype=float) * \
                                      mapping_prior * self.task.n_primitive_actions
        self.mapping_mle = np.ones((self.task.n_ctx, self.task.n_primitive_actions, self.task.n_abstract_actions),
                                   dtype=float) * (1.0/self.task.n_primitive_actions)

        # rewards!
        self.reward_visits = np.ones((self.task.n_ctx, self.task.n_states)) * 0.0001
        self.reward_received = np.ones((self.task.n_ctx, self.task.n_states)) * 0.001
        self.reward_function = np.ones((self.task.n_ctx, self.task.n_states)) * (0.001/0.0101)

    def update(self, experience_tuple):
        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

    def update_rewards(self, c, sp, r):
        self.reward_visits[c, sp] += 1.0
        self.reward_received[c, sp] += r
        self.reward_function[c, sp] = self.reward_received[c, sp] / self.reward_visits[c, sp]

    def updating_mapping(self, c, a, aa):

        self.mapping_history[c, a, aa] += 1.0
        self.abstract_action_counts[c, aa] += 1.0

        for aa0 in range(self.task.n_abstract_actions):
            for a0 in range(self.task.n_primitive_actions):
                self.mapping_mle[c, a0, aa0] = self.mapping_history[c, a0, aa0] / self.abstract_action_counts[c, aa0]

    def select_abstract_action(self, state):

        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            (x, y), c = state
            pi = policy_iteration(self.task.current_trial.transition_function,
                                  self.reward_function[c, :],
                                  gamma=self.gamma,
                                  stop_criterion=self.iteration_criterion)

            #
            s = self.task.state_location_key[(x, y)]
            abstract_action = pi[s]
        else:
            abstract_action = np.random.randint(self.n_abstract_actions)

        return abstract_action

    def select_action(self, state):

        # use epsilon greedy choice function
        if np.random.rand() > self.epsilon:
            _, c = state

            abstract_action = self.select_abstract_action(state)

            pmf = self.mapping_mle[c, :, abstract_action]
            for aa0 in range(self.task.n_abstract_actions):
                if not aa0 == abstract_action:
                    pmf *= (1 - self.mapping_mle[c, :, aa0])

            pmf /= pmf.sum()

            return sample_cmf(pmf.cumsum())
        else:
            return np.random.randint(self.n_primitive_actions)

    def set_reward_prior(self, list_locations):
        """
        This method allows the agent to specific grid coordinates as potential goal locations by
        putting some prior (low confidence) reward density over the grid locations.

        All other locations have low reward probability

        :param list_locations: a list of (x, y) coordinates to consider as priors for the goal location search
        :return: None
        """
        # this makes for a 10% reward received prior over putative non-goal states
        self.reward_visits = np.ones((self.task.n_ctx, self.task.n_states)) * 0.0001
        self.reward_received = np.ones((self.task.n_ctx, self.task.n_states)) * 0.00001

        for loc in list_locations:
            s = self.task.state_location_key[loc]
            self.reward_received[:, s] += 0.001
            self.reward_visits[:, s] += 0.001

        for s in range(self.task.n_states):
            for c in range(self.task.n_ctx):
                self.reward_function[c, s] = self.reward_received[c, s] / self.reward_visits[c, s]


class JointClustering(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0,  discount_rate=0.8, iteration_criterion=0.01,
                 mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        # inverse temperature is used internally by the reward hypothesis to convert q-values into a PMF. We
        # always want a very greedy PMF as this is only used to deal with cases where there are multiple optimal
        # actions
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

        # create task sets, each containing a reward and mapping hypothesis
        # with the same assignment
        self.reward_hypotheses = [RewardHypothesis(
                self.task.n_states, inverse_temperature, discount_rate, iteration_criterion, alpha
            )]
        self.mapping_hypotheses = [MappingHypothesis(
                self.task.n_primitive_actions, self.task.n_abstract_actions, alpha, mapping_prior
            )]

        self.log_belief = np.ones(1, dtype=float)

    def updating_mapping(self, c, a, aa):
        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis
            h_m.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def update(self, experience_tuple):

        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        self.log_belief = np.zeros(len(self.mapping_hypotheses))
        for ii, h_m in enumerate(self.mapping_hypotheses):
            self.log_belief[ii] = h_m.get_log_prior()

        # then update the posterior of the belief distribution with the reward posterior
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis
            self.log_belief[ii] += h_r.get_log_likelihood()

        # then update the posterior of the mappings likelihood (prior is shared, only need it once)
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            self.log_belief[ii] += h_m.get_log_likelihood()

    def augment_assignments(self, context):
        new_reward_hypotheses = list()
        new_mapping_hypotheses = list()
        new_log_belief = list()

        for h_r, h_m in zip(self.reward_hypotheses, self.mapping_hypotheses):
            assert type(h_r) is RewardHypothesis
            assert type(h_m) is MappingHypothesis

            old_assignments = h_r.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_r.deep_copy()
                h_r0.add_new_context_assignment(context, k)

                h_m0 = h_m.deep_copy()
                h_m0.add_new_context_assignment(context, k)

                new_reward_hypotheses.append(h_r0)
                new_mapping_hypotheses.append(h_m0)
                new_log_belief.append(h_r0.get_log_posterior() + h_m0.get_log_likelihood())

        self.reward_hypotheses = new_reward_hypotheses
        self.mapping_hypotheses = new_mapping_hypotheses
        self.log_belief = new_log_belief

    def resample_hypothesis_space(self, n_particles=None):
        if n_particles is None:
            n_particles = self.n_particles
        
        n_beliefs = len(self.log_belief)
        if n_beliefs > n_particles:
            old_reward_hypotheses = np.array(self.reward_hypotheses)
            old_mapping_hypotheses = np.array(self.mapping_hypotheses)
            old_log_beliefs = np.array(self.log_belief)
            
            # first select the n_max MAP hypotheses
            arg_idx = np.argpartition(old_log_beliefs, -self.n_max)[-self.n_max:]
            new_reward_hypotheses = old_reward_hypotheses[arg_idx]
            new_mapping_hypotheses = old_mapping_hypotheses[arg_idx]
            new_log_beliefs = old_log_beliefs[arg_idx]
            
            # select out remaining elements
            arg_idx = np.isin(range(n_beliefs),arg_idx[-self.n_max:], invert=True)
            old_reward_hypotheses = old_reward_hypotheses[arg_idx]
            old_mapping_hypotheses = old_mapping_hypotheses[arg_idx]
            old_log_beliefs = old_log_beliefs[arg_idx]
            
            # randomly sample remaining elements according to their posterior
            p = np.exp(old_log_beliefs*self.sample_tau)
            p = p/np.sum(p)
            arg_idx = np.random.choice(n_beliefs-self.n_max, n_particles-self.n_max, p=p, replace=False)
            new_reward_hypotheses = np.concatenate((new_reward_hypotheses, old_reward_hypotheses[arg_idx]))
            new_mapping_hypotheses = np.concatenate((new_mapping_hypotheses, old_mapping_hypotheses[arg_idx]))
            new_log_beliefs = np.concatenate((new_log_beliefs, old_log_beliefs[arg_idx]))
            
            self.log_belief = new_log_beliefs
            self.reward_hypotheses = new_reward_hypotheses
            self.mapping_hypotheses = new_mapping_hypotheses
            
    def select_abstract_action(self, state):
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief)
        h_r = self.reward_hypotheses[ii]
        q_values = h_r.select_abstract_action_pmf(s, c, self.task.current_trial.transition_function)

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def select_action(self, state):
        # use softmax choice function
        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief)
        h_m = self.mapping_hypotheses[ii]

        mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

        return sample_cmf(mapping_mle.cumsum())

    def get_reward_function(self, state):
        # Get the q-values over abstract actions
        _, c = state

        ii = np.argmax(self.log_belief)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_function(c)

    def get_reward_prediction(self, x, y, c):
        sp = self.task.state_location_key[(x, y)]
        ii = np.argmax(self.log_belief)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_prediction(c, sp)

class IndependentClusterAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0, discount_rate=0.8,
                 iteration_criterion=0.01,
                 mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

        # get the list of enumerated set assignments!

        # create task sets, each containing a reward and mapping hypothesis
        # with the same assignment
        self.reward_hypotheses = [
            RewardHypothesis(
                self.task.n_states, inverse_temperature, discount_rate,
                iteration_criterion, alpha
            )]
        self.mapping_hypotheses = [
            MappingHypothesis(
                self.task.n_primitive_actions, self.task.n_abstract_actions,
                alpha, mapping_prior
            )]

        self.log_belief_rew = np.ones(1, dtype=float)
        self.log_belief_map = np.ones(1, dtype=float)

    def updating_mapping(self, c, a, aa):
        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis
            h_m.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def update(self, experience_tuple):

        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        # then update the posterior of the rewards
        for ii, h_r in enumerate(self.reward_hypotheses):
            assert type(h_r) is RewardHypothesis
            self.log_belief_rew[ii] = h_r.get_log_posterior()

        # then update the posterior of the mappings
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            self.log_belief_map[ii] = h_m.get_log_posterior()

    def augment_assignments(self, context):
        new_hypotheses = list()
        new_log_belief = list()

        for h_r in self.reward_hypotheses:
            assert type(h_r) is RewardHypothesis

            old_assignments = h_r.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_r.deep_copy()
                h_r0.add_new_context_assignment(context, k)

                new_hypotheses.append(h_r0)
                new_log_belief.append(h_r0.get_log_prior() + h_r0.get_log_likelihood())

        self.reward_hypotheses = new_hypotheses
        self.log_belief_rew = new_log_belief

        new_hypotheses = list()
        new_log_belief = list()

        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis

            old_assignments = h_m.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_m0 = h_m.deep_copy()
                h_m0.add_new_context_assignment(context, k)

                new_hypotheses.append(h_m0)
                new_log_belief.append(h_m0.get_log_prior() + h_m0.get_log_likelihood())

        self.mapping_hypotheses = new_hypotheses
        self.log_belief_map = new_log_belief

    def resample_hypothesis_space(self, n_particles=None):
        self.log_belief_rew, self.reward_hypotheses = self.resample_hypotheses(self.reward_hypotheses, 
                                    self.log_belief_rew, n_particles)
        self.log_belief_map, self.mapping_hypotheses = self.resample_hypotheses(self.mapping_hypotheses, 
                                    self.log_belief_map, n_particles)

    def select_abstract_action(self, state):
        # use softmax greedy choice function
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief_rew)
        h_r = self.reward_hypotheses[ii]

        q_values = h_r.select_abstract_action_pmf(
            s, c, self.task.current_trial.transition_function
        )

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def get_reward_function(self, state):
        _, c = state

        ii = np.argmax(self.log_belief_rew)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_function(c)

    def select_action(self, state):
        # use softmax greedy choice function
        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief_map)
        h_m = self.mapping_hypotheses[ii]

        mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

        return sample_cmf(mapping_mle.cumsum())

    def get_reward_prediction(self, x, y, c):
        sp = self.task.state_location_key[(x, y)]
        ii = np.argmax(self.log_belief_rew)
        h_r = self.reward_hypotheses[ii]
        return h_r.get_reward_prediction(c, sp)


class MetaAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0, discount_rate=0.8,
                 iteration_criterion=0.01,
                 mapping_prior=0.01, m_biases=[0.0, 0.0]):
        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.independent_agent = IndependentClusterAgent(
            task, inverse_temperature=inverse_temperature,  alpha=alpha, discount_rate=discount_rate,
                 iteration_criterion=iteration_criterion, mapping_prior=mapping_prior
        )
        self.joint_agent = JointClustering(
            task, inverse_temperature=inverse_temperature,  alpha=alpha, discount_rate=discount_rate,
                 iteration_criterion=iteration_criterion, mapping_prior=mapping_prior
        )

        self.responsibilities = {'Ind': m_biases[0], 'Joint': m_biases[0]}
        # self.responsibilities = {'Ind': 0.5 + mix_bias, 'Joint': 0.5 - mix_bias}
        # self.eta = mixing_lrate
        # self.beta = mixing_temp
        self.is_mixture = True

        self.choose_operating_model()

        # self.current_agent = self.independent_agent
        # self.current_agent_name = 'Ind'
        # if np.random.rand() < 0.5:
        #     self.current_agent = self.joint_agent
        #     self.current_agent_name = 'Joint'

    def is_meta(self):
        return True

    def get_joint_probability(self):
        # k = np.sum(np.exp(self.beta * np.array(self.responsibilities.values())))
        # return np.exp(self.beta * self.responsibilities['Joint']) / k
        return np.exp(self.responsibilities['Joint'] - logsumexp(self.responsibilities.values()))


    def choose_operating_model(self):
        if np.random.rand() < self.get_joint_probability():
            self.current_agent = self.joint_agent
            self.current_agent_name = 'Joint'
        else:
            self.current_agent = self.independent_agent
            self.current_agent_name = 'Ind'


    def update(self, experience_tuple):
        self.independent_agent.update(experience_tuple)
        self.joint_agent.update(experience_tuple)

    def new_trial_function(self):
        self.choose_operating_model()

    def augment_assignments(self, context):
        self.independent_agent.augment_assignments(context)
        self.joint_agent.augment_assignments(context)

    def resample_hypothesis_space(self, n_particles=None):
        self.joint_agent.resample_hypothesis_space(n_particles)
        self.independent_agent.resample_hypothesis_space(n_particles)

    def select_action(self, state):
        self.choose_operating_model()
        return self.current_agent.select_action(state)
        
    def evaluate_mixing_agent(self, xp, yp, c, r):
        # get the reward prediction for the MAP joint and MAP ind hypotheses
        r_hat_i = self.independent_agent.get_reward_prediction(xp, yp, c)
        r_hat_j = self.joint_agent.get_reward_prediction(xp, yp, c)

        # The map estimate is sensitive to underflow error -- this prevents this be assuming the
        # model has some probability it is wrong (here, hard coded as 1/1000) and bounding the
        # models' probability estimates of reward
        r_hat_j = np.max([0.999 * r_hat_j, 0.001])
        r_hat_i = np.max([0.999 * r_hat_i, 0.001])

        # what is the predicted probability of the observed output for each model? Track the log prob
        self.responsibilities['Joint'] += np.log(r * r_hat_j + (1 - r) * (1 - r_hat_j))
        self.responsibilities['Ind']   += np.log(r * r_hat_i + (1 - r) * (1 - r_hat_i))
        # when r==1, returns the probability of reward; when r==0, return the probability of no reward


class FlatControlAgent(ModelBasedAgent):

    def __init__(self, task, inverse_temperature=100.0, alpha=1.0,
                 discount_rate=0.8, iteration_criterion=0.01,
                 mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        # inverse temperature is used internally by the reward hypothesis to convert
        # q-values into a PMF. We
        # always want a very greedy PMF as this is only used to deal with cases where
        # there are multiple optimal
        # actions
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

        # create task sets, each containing a reward and mapping hypothesis
        # with the same assignment
        self.task_sets = [{
                'Reward Hypothesis': RewardHypothesis(
                    self.task.n_states, inverse_temperature, discount_rate,
                    iteration_criterion, alpha
                ),
                'Mapping Hypothesis': MappingHypothesis(
                    self.task.n_primitive_actions, self.task.n_abstract_actions,
                    alpha, mapping_prior
                ),
            }]

        self.log_belief = np.ones(1)

    def updating_mapping(self, c, a, aa):
        for ts in self.task_sets:
            h_m = ts['Mapping Hypothesis']
            assert type(h_m) is MappingHypothesis
            h_m.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for ts in self.task_sets:
            h_r = ts['Reward Hypothesis']
            assert type(h_r) is RewardHypothesis
            h_r.update(c, sp, r)

    def update(self, experience_tuple):

        # super(FlatAgent, self).update(experience_tuple)
        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        # then update the posterior
        for ii, ts in enumerate(self.task_sets):
            h_m = ts['Mapping Hypothesis']
            h_r = ts['Reward Hypothesis']

            assert type(h_m) is MappingHypothesis
            assert type(h_r) is RewardHypothesis

            self.log_belief[ii] = h_m.get_log_prior() + \
                h_m.get_log_likelihood() + h_r.get_log_likelihood()

    def augment_assignments(self, context):

        ts = self.task_sets[0]
        h_m = ts['Mapping Hypothesis']
        h_r = ts['Reward Hypothesis']
        assert type(h_m) is MappingHypothesis
        assert type(h_r) is RewardHypothesis

        h_m.add_new_context_assignment(context, context)
        h_r.add_new_context_assignment(context, context)

        self.task_sets = [{'Reward Hypothesis': h_r, 'Mapping Hypothesis': h_m}]
        self.log_belief = [1]

    def select_abstract_action(self, state):
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief)
        h_r = self.task_sets[ii]['Reward Hypothesis']
        q_values = h_r.select_abstract_action_pmf(
            s, c, self.task.current_trial.transition_function
        )

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def select_action(self, state):

        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief)
        h_m = self.task_sets[ii]['Mapping Hypothesis']

        mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            mapping_mle[a0] = h_m.get_mapping_probability(c, a0, aa)

        return sample_cmf(mapping_mle.cumsum())

    def get_reward_function(self, state):
        # Get the q-values over abstract actions
        _, c = state

        ii = np.argmax(self.log_belief)
        h_r = self.task_sets[ii]['Reward Hypothesis']
        return h_r.get_reward_function(c)


class HierarchicalAgent(ModelBasedAgent):
    def __init__(self, task, inverse_temperature=100.0, alpha0=1.0, alpha1=0.5, discount_rate=0.8,
                 iteration_criterion=0.01,
                 mapping_prior=0.01):

        assert type(task) is Task
        super(FullInformationAgent, self).__init__(task)

        self.inverse_temperature = inverse_temperature
        self.gamma = discount_rate
        self.iteration_criterion = iteration_criterion
        self.current_trial = 0
        self.n_abstract_actions = self.task.n_abstract_actions
        self.n_primitive_actions = self.task.n_primitive_actions

        # get the list of enumerated set assignments!
        
        self.rooms_hypotheses = [HierarchicalHypothesis(self.task.n_states, 
                    self.n_primitive_actions, self.n_abstract_actions, self.inverse_temperature, 
                    self.gamma, self.iteration_criterion, alpha0, alpha1, mapping_prior)]
    
        self.log_belief = np.ones(1, dtype=float)

    def updating_mapping(self, c, a, aa):
        for h_room in self.rooms_hypotheses:
            assert type(h_room) is HierarchicalHypothesis
            h_room.updating_mapping(c, a, aa)

    def update_rewards(self, c, sp, r):
        for h_room in self.rooms_hypotheses:
            assert type(h_room) is HierarchicalHypothesis
            h_room.update(c, sp, r)

    def update(self, experience_tuple):

        _, a, aa, r, (loc_prime, c) = experience_tuple
        self.updating_mapping(c, a, aa)
        sp = self.task.state_location_key[loc_prime]
        self.update_rewards(c, sp, r)

        # then update the posterior
        for ii, h_room in enumerate(self.rooms_hypotheses):
            assert type(h_room) is HierarchicalHypothesis
            self.log_belief[ii] = h_room.get_log_posterior()


    def augment_assignments(self, context):
        # first augment room assignments
        new_hypotheses0 = list()
        for h_room in self.rooms_hypotheses:
            assert type(h_room) is HierarchicalHypothesis

            old_assignments = h_room.get_room_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_room.deep_copy()
                h_r0.add_new_room_context_assignment(context, k)

                new_hypotheses0.append(h_r0)
                
        # next augment reward assignments
        new_hypotheses1 = list()
        for h_room in new_hypotheses0:
            assert type(h_room) is HierarchicalHypothesis

            old_assignments = h_room.get_reward_assignments(context)
            new_assignments = augment_hierarchical_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                h_r0 = h_room.deep_copy()
                h_r0.add_new_reward_context_assignment(context, assignment)
                new_hypotheses1.append(h_r0)
        
        # finally augment clustering assignments
        new_hypotheses2 = list()
        new_log_belief = list()
        for h_room in new_hypotheses1:
            assert type(h_room) is HierarchicalHypothesis

            old_assignments = h_room.get_mapping_assignments(context)
            new_assignments = augment_hierarchical_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                h_r0 = h_room.deep_copy()
                h_r0.add_new_mapping_context_assignment(context, assignment)
                new_hypotheses2.append(h_r0)
                new_log_belief.append(h_r0.get_log_posterior())

        self.rooms_hypotheses = new_hypotheses2
        self.log_belief = new_log_belief

    def resample_hypothesis_space(self, n_particles=None):
        self.log_belief, self.rooms_hypotheses = self.resample_hypotheses(self.rooms_hypotheses, 
                                    self.log_belief, n_particles)

    def select_abstract_action(self, state):
        # use softmax greedy choice function
        (x, y), c = state
        s = self.task.state_location_key[(x, y)]

        ii = np.argmax(self.log_belief)
        h_room = self.rooms_hypotheses[ii]

        q_values = h_room.select_abstract_action_pmf(
            s, c, self.task.current_trial.transition_function
        )

        full_pmf = np.exp(q_values * self.inverse_temperature)
        full_pmf = full_pmf / np.sum(full_pmf)

        return sample_cmf(full_pmf.cumsum())

    def get_reward_function(self, state):
        _, c = state

        ii = np.argmax(self.log_belief)
        h_room = self.rooms_hypotheses[ii]
        return h_room.get_reward_function(c)

    def select_action(self, state):
        # use softmax greedy choice function
        _, c = state
        aa = self.select_abstract_action(state)
        c = np.int32(c)

        ii = np.argmax(self.log_belief)
        h_room = self.rooms_hypotheses[ii]

        mapping_mle = np.zeros(self.n_primitive_actions)
        for a0 in np.arange(self.n_primitive_actions, dtype=np.int32):
            mapping_mle[a0] = h_room.get_mapping_probability(c, a0, aa)

        return sample_cmf(mapping_mle.cumsum())

    def get_reward_prediction(self, x, y, c):
        sp = self.task.state_location_key[(x, y)]
        ii = np.argmax(self.log_belief)
        h_room = self.rooms_hypotheses[ii]
        return h_room.get_reward_prediction(c, sp)