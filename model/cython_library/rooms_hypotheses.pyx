# cython: profile=True, linetrace=True, boundscheck=True, wraparound=True
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from core import get_prior_log_probability

DTYPE = np.float
ctypedef np.float_t DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT_DTYPE_t

cdef extern from "math.h":
    double log(double x)

cdef class Hypothesis(object):

    cdef double get_log_posterior(self):
        pass

cdef class MappingCluster(object):
    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a
    cdef double [:] abstract_action_counts, primitive_action_counts
    cdef int n_primitive_actions, n_abstract_actions
    cdef double mapping_prior

    def __init__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):

        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a
        cdef double[:] abstract_action_counts, primitive_action_counts

        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=float) * mapping_prior
        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=float) *  mapping_prior * n_primitive_actions
        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=float) * \
                      (1.0 / n_primitive_actions)

        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE) * mapping_prior * n_abstract_actions
        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE) * \
                        (1.0 / n_abstract_actions)

        self.mapping_history = mapping_history
        self.abstract_action_counts = abstract_action_counts
        self.mapping_mle = mapping_mle
        self.primitive_action_counts = primitive_action_counts
        self.pr_aa_given_a = pr_aa_given_a

        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.mapping_prior = mapping_prior

    def update(self, int a, int aa):
        cdef int aa0, a0
        self.mapping_history[a, aa] += 1.0
        self.abstract_action_counts[aa] += 1.0
        self.primitive_action_counts[a] += 1.0

        for aa0 in range(self.n_abstract_actions):
            for a0 in range(self.n_primitive_actions):
                self.mapping_mle[a0, aa0] = self.mapping_history[a0, aa0] / self.abstract_action_counts[aa0]

                # p(A|a, k) estimator
                self.pr_aa_given_a[a0, aa0] = self.mapping_history[a0, aa0] / self.primitive_action_counts[a0]

    def get_mapping_mle(self, int a, int aa):
        return self.mapping_mle[a, aa]

    def get_likelihood(self, int a, int aa):
        return self.pr_aa_given_a[a, aa]

    def deep_copy(self):
        cdef int a, aa, idx

        cdef MappingCluster _cluster_copy = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                                           self.mapping_prior)

        _cluster_copy.primitive_action_counts = np.copy(self.primitive_action_counts)
        _cluster_copy.mapping_history = np.copy(self.mapping_history)
        _cluster_copy.mapping_mle = np.copy(self.mapping_mle)
        _cluster_copy.pr_aa_given_a = np.copy(self.pr_aa_given_a)
        _cluster_copy.abstract_action_counts = np.copy(self.abstract_action_counts)

#        for a in range(self.n_primitive_actions):
#            _cluster_copy.primitive_action_counts[a] = self.primitive_action_counts[a]
#
#            for aa in range(self.n_abstract_actions + 1): # include the possibility of the "wait" action
#                _cluster_copy.mapping_history[a, aa] = self.mapping_history[a, aa]
#                _cluster_copy.mapping_mle[a, aa] = self.mapping_mle[a, aa]
#                _cluster_copy.pr_aa_given_a[a, aa] = self.pr_aa_given_a[a, aa]
#
#        for aa in range(self.n_abstract_actions + 1): # include the possibility of the "wait" action
#            _cluster_copy.abstract_action_counts[aa] = self.abstract_action_counts[aa]

        return _cluster_copy


cdef class MappingHypothesis(object):

    cdef dict cluster_assignments, clusters
    cdef double prior_log_prob, alpha, mapping_prior
    cdef list experience
    cdef int n_abstract_actions, n_primitive_actions

    def __init__(self, int n_primitive_actions, int n_abstract_actions,
                 float alpha, float mapping_prior):

        self.cluster_assignments = dict()
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.alpha = alpha
        self.mapping_prior = mapping_prior

        # initialize the clusters
        self.clusters = dict()

        # store the prior probability
        self.prior_log_prob = 1.0

        # need to store all experiences for log probability calculations
        self.experience = list()

    def update_mapping(self, int c, int a, int aa):
        cdef int k = self.cluster_assignments[c]
        cdef MappingCluster cluster = self.clusters[k]
        cluster.update(a, aa)
        self.clusters[k] = cluster

        # need to store all experiences for log probability calculations
        self.experience.append((k, a, aa))

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, a, aa
        cdef MappingCluster cluster

        #loop through experiences and get posterior
        for k, a, aa in self.experience:
            cluster = self.clusters[k]
            log_likelihood += log(cluster.get_likelihood(a, aa))

        return log_likelihood

    def get_obs_likelihood(self, int c, int a, int aa):
        cdef int k = self.cluster_assignments[c]
        cdef MappingCluster cluster = self.clusters[k]
        return cluster.get_likelihood(a, aa)

    def get_log_posterior(self):
        return self.prior_log_prob + self.get_log_likelihood()

    def get_mapping_probability(self, int c, int a, int aa):
        cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
        return cluster.get_mapping_mle(a, aa)

    def get_log_prior(self):
        return self.prior_log_prob

    def deep_copy(self):
        cdef MappingHypothesis _h_copy = MappingHypothesis(
            self.n_primitive_actions, self.n_abstract_actions, self.alpha,
            self.mapping_prior
        )

        cdef int k, a, aa
        cdef MappingCluster cluster

        _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.experience = [(k, a, aa) for k, a, aa in self.experience]
        _h_copy.prior_log_prob = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add a new mapping cluster
            self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)

        self.cluster_assignments[c] = k
        self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)



cdef class GoalCluster(object):
    cdef double set_visits, goal_prior
    cdef double [:]  goal_rewards_received, goal_reward_probability
    cdef int n_goals

    def __init__(self, int n_goals, float goal_prior):
        self.n_goals = n_goals
        self.goal_prior = goal_prior

        # rewards!
        self.set_visits =  n_goals * goal_prior
        self.goal_rewards_received = np.ones(n_goals) * goal_prior
        self.goal_reward_probability = np.ones(n_goals) * (1.0 / n_goals)

    def update(self, int goal, int r):
        cdef double r0
        cdef int g0

        self.set_visits += 1.0
        self.goal_rewards_received[goal] += r

        if r == 0:
            r0 = 1.0 / (self.n_goals - 1)
            for g0 in range(self.n_goals):
                if g0 != goal:
                    self.goal_rewards_received[g0] += r0

        # update all goal probabilities
        for g0 in range(self.n_goals):
            self.goal_reward_probability[g0] = self.goal_rewards_received[g0] / self.set_visits

    def get_observation_probability(self, int goal, int r):
        if r == 0:
            return 1 - self.goal_reward_probability[goal]
        return self.goal_reward_probability[goal]

    def get_goal_pmf(self):
        return self.goal_reward_probability


    def deep_copy(self):
        cdef int g

        cdef GoalCluster _cluster_copy = GoalCluster(self.n_goals, self.goal_prior)

        _cluster_copy.set_visits = self.set_visits
        _cluster_copy.goal_rewards_received = np.copy(self.goal_rewards_received)
        _cluster_copy.goal_reward_probability = np.copy(self.goal_reward_probability)

#        for g in range(self.n_goals):
#            _cluster_copy.set_visits = self.set_visits
#            _cluster_copy.goal_rewards_received[g] = self.goal_rewards_received[g]
#            _cluster_copy.goal_reward_probability[g] = self.goal_reward_probability[g]

        return _cluster_copy


cdef class GoalHypothesis(object):
    cdef int n_goals
    cdef double log_prior, alpha, goal_prior
    cdef dict cluster_assignments, clusters
    cdef list experience

    def __init__(self, int n_goals, float alpha, float goal_prior):

        self.n_goals = n_goals
        self.cluster_assignments = dict()
        self.alpha = alpha
        self.goal_prior = goal_prior

        # initialize goal clusters
        self.clusters = dict()

        # initialize posterior
        self.experience = list()
        self.log_prior = 1.0

    def update(self, int c, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        cluster.update(goal, r)
        self.clusters[k] = cluster

        self.experience.append((k, goal, r))

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, goal, r
        cdef GoalCluster cluster

        #loop through experiences and get posterior
        for k, goal, r in self.experience:
            cluster = self.clusters[k]
            log_likelihood += log(cluster.get_observation_probability(goal, r))

        return log_likelihood

    def get_obs_likelihood(self, int c, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        return cluster.get_observation_probability(goal, r)

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.log_prior

    def get_log_prior(self):
        return self.log_prior

    def get_goal_probability(self, int c):
        cdef int g, k

        k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)

        cdef double [:] rew_func = cluster.get_goal_pmf()
        goal_probability = np.copy(rew_func)
#        for g in range(self.n_goals):
#            goal_probability[g] = rew_func[g]
        return goal_probability


    def deep_copy(self):
        cdef GoalHypothesis _h_copy = GoalHypothesis(self.n_goals, self.alpha, self.goal_prior)

        cdef int k, goal, r
        cdef GoalCluster cluster

        _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.experience = [(k, goal, r) for k, goal, r in self.experience]
        _h_copy.log_prior = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = GoalCluster(self.n_goals, self.goal_prior)

        self.cluster_assignments[c] = k  # note, there's no check built in here
        self.log_prior = get_prior_log_probability(self.cluster_assignments, self.alpha)
        
        
cdef class SublvlGoalHypothesis(object):
    cdef int n_goals, n_sublvls
    cdef double log_prior, alpha, goal_prior
    cdef list clusters, cluster_assignments, experience
    cdef double[:] _sublvl_log_prior

    def __init__(self, int n_sublvls, int n_goals, float alpha, float goal_prior):
        cdef int i
        
        self.n_sublvls = n_sublvls
        self.n_goals = n_goals
        self.alpha = alpha
        self.goal_prior = goal_prior

        # initialize posterior
        self.experience = list()
        self.log_prior = 1.0

        # initialize goal clusters and posterior, one for each sublvl
        self.cluster_assignments = []
        self.clusters = [dict() for i in range(n_sublvls)]
            
        self._sublvl_log_prior = np.ones(n_sublvls)*1.0/n_sublvls

    def update(self, int c, int goal, int r):
        cdef list sublvl_list = [ ii for ii, _subassignment in enumerate(self.cluster_assignments) if c in _subassignment.keys() ]
        assert len(sublvl_list) == 1
        cdef int sublvl = sublvl_list[0]
        cdef int k = self.cluster_assignments[sublvl][c]

        cdef GoalCluster cluster = self.clusters[sublvl][k]
        cluster.update(goal, r)
        self.clusters[sublvl][k] = cluster

        self.experience.append((sublvl, k, goal, r))

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int sublvl, k, goal, r
        cdef GoalCluster cluster

        #loop through experiences and get posterior
        for sublvl, k, goal, r in self.experience:
            cluster = self.clusters[sublvl][k]
            log_likelihood += log(cluster.get_observation_probability(goal, r))

        return log_likelihood
    
    def get_obs_likelihood(self, int c, int goal, int r):
        cdef list sublvl_list = [ ii for ii, _subassignment in enumerate(self.cluster_assignments) if c in _subassignment.keys() ]
        assert len(sublvl_list) == 1
        cdef int sublvl = sublvl_list[0]
        cdef int k = self.cluster_assignments[sublvl][c]

        cdef GoalCluster cluster = self.clusters[sublvl][k]
        return cluster.get_observation_probability(goal, r)

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.log_prior

    def get_log_prior(self):
        return self.log_prior

    def get_goal_probability(self, int c):
        cdef list sublvl_list = [ ii for ii, _subassignment in enumerate(self.cluster_assignments) if c in _subassignment.keys()]
        assert len(sublvl_list) == 1
        cdef int sublvl = sublvl_list[0]
        cdef int k = self.cluster_assignments[sublvl][c]
        
        cdef GoalCluster cluster = self.clusters[sublvl][k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)

        cdef int g
        cdef double [:] rew_func = cluster.get_goal_pmf()
        goal_probability = np.copy(rew_func)
#        for g in range(self.n_goals):
#            goal_probability[g] = rew_func[g]
        return goal_probability

    def deep_copy(self):
        cdef SublvlGoalHypothesis _h_copy = SublvlGoalHypothesis(self.n_sublvls, self.n_goals, self.alpha, self.goal_prior)

        cdef int sublvl, k, goal, r, ii
        cdef GoalCluster cluster

        _h_copy.clusters = [{k: cluster.deep_copy() for k, cluster in self.clusters[ii].iteritems()} for ii in range(self.n_sublvls)]
        _h_copy.cluster_assignments = [{c: k for c, k in self.cluster_assignments[ii].iteritems()} for ii in range(len(self.cluster_assignments))]
        _h_copy._sublvl_log_prior = np.copy(self._sublvl_log_prior)
        _h_copy.experience = [(sublvl, k, goal, r) for sublvl, k, goal, r in self.experience]
        _h_copy.log_prior = np.sum(_h_copy._sublvl_log_prior)

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int sublvl, int c, int k):
        cdef int n_subassignments = len(self.cluster_assignments)
        if sublvl >= n_subassignments:
            assert sublvl == n_subassignments and n_subassignments < self.n_sublvls
               # first condition ensures sublvl indexes element
               # that will be appended to self.cluster_assignments
            self.cluster_assignments.append(dict())
        
        # note: these are shallow copies (ie. aliases)
        cdef dict subassignments = self.cluster_assignments[sublvl]
        cdef dict sub_clusters = self.clusters[sublvl]
        
        # check if cluster "k" is already been assigned new cluster
        if k not in subassignments.values():
            # if not, add an new reward cluster
            sub_clusters[k] = GoalCluster(self.n_goals, self.goal_prior)

        subassignments[c] = k  # note, there's no check built in here
        self._sublvl_log_prior[sublvl] = get_prior_log_probability(subassignments, self.alpha)
        self.log_prior = np.sum(self._sublvl_log_prior)


cdef class SublvlMappingHypothesis(object):

    cdef list cluster_assignments, clusters, experience
    cdef double prior_log_prob, alpha, mapping_prior
    cdef int n_abstract_actions, n_primitive_actions, n_sublvls
    cdef double[:] _sublvl_log_prior

    def __init__(self, int n_sublvls, int n_primitive_actions, int n_abstract_actions,
                 float alpha, float mapping_prior):
        cdef int i

        self.n_sublvls = n_sublvls
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.alpha = alpha
        self.mapping_prior = mapping_prior
        
        # store the prior probability
        self.prior_log_prob = 1.0

        # initialize goal clusters and posterior, one for each sublvl
        self.clusters = [dict() for ii in range(n_sublvls)]
        self.cluster_assignments = []
        
        self._sublvl_log_prior = np.ones(n_sublvls)*1.0/n_sublvls

        # need to store all experiences for log probability calculations
        self.experience = list()
        
    def update_mapping(self, int c, int a, int aa):
        cdef list sublvl_list = [ ii for ii, _subassignment in enumerate(self.cluster_assignments) if c in _subassignment.keys()]
        assert len(sublvl_list) == 1
        cdef int sublvl = sublvl_list[0]
        cdef int k = self.cluster_assignments[sublvl][c]

        cdef MappingCluster cluster = self.clusters[sublvl][k]
        cluster.update(a, aa)
        self.clusters[sublvl][k] = cluster

        # need to store all experiences for log probability calculations
        self.experience.append((sublvl, k, a, aa))

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int sublvl, k, a, aa
        cdef MappingCluster cluster

        #loop through experiences and get posterior
        for sublvl, k, a, aa in self.experience:
            cluster = self.clusters[sublvl][k]
            log_likelihood += log(cluster.get_likelihood(a, aa))

        return log_likelihood
    
    def get_obs_likelihood(self, int c, int a, int aa):
        cdef list sublvl_list = [ ii for ii, _subassignment in enumerate(self.cluster_assignments) if c in _subassignment.keys() ]
        assert len(sublvl_list) == 1
        cdef int sublvl = sublvl_list[0]
        cdef int k = self.cluster_assignments[sublvl][c]

        cdef MappingCluster cluster = self.clusters[sublvl][k]
        return cluster.get_likelihood(a, aa)

    def get_log_posterior(self):
        return self.prior_log_prob + self.get_log_likelihood()

    def get_mapping_probability(self, int c, int a, int aa):
        cdef list sublvl_list = [ ii for ii, _subassignment in enumerate(self.cluster_assignments) if c in _subassignment.keys()]
        assert len(sublvl_list) == 1
        cdef int sublvl = sublvl_list[0]
        cdef int k = self.cluster_assignments[sublvl][c]
        
        cdef MappingCluster cluster = self.clusters[sublvl][k]
        return cluster.get_mapping_mle(a, aa)

    def get_log_prior(self):
        return self.prior_log_prob

    def deep_copy(self):
        cdef SublvlMappingHypothesis _h_copy = SublvlMappingHypothesis(
            self.n_sublvls, self.n_primitive_actions, self.n_abstract_actions,
            self.alpha, self.mapping_prior
        )

        cdef int sublvl, k, a, aa, ii
        cdef MappingCluster cluster

        _h_copy.clusters = [{k: cluster.deep_copy() for k, cluster in self.clusters[ii].iteritems()} for ii in range(self.n_sublvls)]
        _h_copy.cluster_assignments = [{c: k for c, k in self.cluster_assignments[ii].iteritems()} for ii in range(len(self.cluster_assignments))]
        for ii in range(len(self.cluster_assignments)):
            _h_copy._sublvl_log_prior[ii] = get_prior_log_probability(_h_copy.cluster_assignments[ii], _h_copy.alpha)
            
        _h_copy.experience = [(sublvl, k, a, aa) for sublvl, k, a, aa in self.experience]
        _h_copy.prior_log_prob = np.sum(_h_copy._sublvl_log_prior)

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int sublvl, int c, int k):
        cdef int n_sublvl_CRPs = len(self.cluster_assignments)
        if sublvl >= n_sublvl_CRPs:
            assert sublvl == n_sublvl_CRPs and n_sublvl_CRPs < self.n_sublvls
               # first condition ensures sublvl indexes element
               # that will be appended to self.cluster_assignments
            self.cluster_assignments.append(dict())
        
        # note: these are shallow copies (ie. aliases)
        cdef dict subassignments = self.cluster_assignments[sublvl]
        cdef dict sub_clusters = self.clusters[sublvl]
        
        # check if cluster "k" is already been assigned new cluster
        if k not in subassignments.values():
            # if not, add a new mapping cluster
            sub_clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)

        subassignments[c] = k  # note, there's no check built in here
        self._sublvl_log_prior[sublvl] = get_prior_log_probability(subassignments, self.alpha)
        self.prior_log_prob = np.sum(self._sublvl_log_prior)


cdef class UpperDoorCluster(object):
    cdef int n_goals
    cdef double set_visits, goal_prior
    cdef double [:,:]  goal_rewards_received, goal_reward_probability

    def __init__(self, int n_goals, float goal_prior):
        self.n_goals = n_goals
        self.goal_prior = goal_prior

        # rewards!
        self.set_visits =  n_goals * goal_prior
        self.goal_rewards_received = np.ones((n_goals,n_goals)) * goal_prior
        self.goal_reward_probability = np.ones((n_goals,n_goals)) * (1.0 / n_goals)
        # agent must visit all doors in correct sequence
        # nth row of goal_reward_probability reward prob distribution if heading for nth door in sequence
        # similarly for goal_rewards_received


#    def update(self, int seq, int goal, int r):
#        cdef double r0, r1
#        cdef int s0, g0
#
#        self.set_visits += 1.0
#        r0 = 1.0 / (self.n_goals - 1)
#
#        if r > 0:     
#            self.goal_rewards_received[seq ,goal] += r
#            
#            for s0 in range(self.n_goals):
#                if s0 != goal:
#                    for g0 in range(self.n_goals):
#                        if g0 != goal:
#                            self.goal_rewards_received[s0, g0] += r0
#        else:
#            r1 = (self.n_goals - 2.0)/(self.n_goals - 1)**2
#            for s0 in range(self.n_goals):
#                for g0 in range(self.n_goals):
#                    if s0 != goal and g0 != goal:
#                        self.goal_rewards_received[s0, g0] += r1
#                    elif not (s0 == goal and g0 == goal):
#                        self.goal_rewards_received[s0, g0] += r0
#
#        # update all goal probabilities
#        for s0 in range(self.n_goals):
#            for g0 in range(self.n_goals):
#                self.goal_reward_probability[s0,g0] = self.goal_rewards_received[s0,g0] / self.set_visits
                


    def update(self, int seq, int goal, int r):
        cdef double r0, r1, baseline
        cdef int s0, g0, count
        cdef double [:,:] probability_subset

        self.set_visits += 1.0
        
        baseline = 1.0 / (self.n_goals+1)

        if r > 0:     
            self.goal_reward_probability[seq+1:self.n_goals,goal] = 0
            self.goal_reward_probability[seq,:] = 0
            self.goal_reward_probability[seq,goal] = 1
            
            for s0 in range(seq+1,self.n_goals):
                norm = np.sum(self.goal_reward_probability[s0,:])
                for g0 in range(self.n_goals):
                    self.goal_reward_probability[s0,g0] /= norm
        else:
            assert self.goal_reward_probability[seq,goal] < 1.0
            self.goal_reward_probability[seq,goal] = 0
            
            # remaining unvisited goals for current seq has equal probability of being true goal
            count = 0
            
            for g0 in range(self.n_goals):
                if self.goal_reward_probability[seq,g0] > baseline:
                    # count non-zero entries of given sequence.
                    # Anything below baseline should be regarded as zero.
                    count += 1
                    
            r0 = 1.0 / count
            
            for g0 in range(self.n_goals):
                if 1.0 > self.goal_reward_probability[seq,g0] and self.goal_reward_probability[seq,g0] > baseline:
                    # condition overwrites only probabilities that are 1 or 0 
                    self.goal_reward_probability[seq,g0] = r0
                    
            # note that reward probabilities for goals in s > seq will get updated
            # when agent finds correct door.


    def get_observation_probability(self, int seq, int goal, int r):
        if r == 0:
            return 1 - self.goal_reward_probability[seq,goal]
        return self.goal_reward_probability[seq,goal]

    def get_goal_pmf(self, int seq):
        return self.goal_reward_probability[seq,:]

    def deep_copy(self):
        cdef int g0, g1

        cdef UpperDoorCluster _cluster_copy = UpperDoorCluster(self.n_goals, self.goal_prior)
        
        _cluster_copy.set_visits = self.set_visits
        _cluster_copy.goal_rewards_received = np.copy(self.goal_rewards_received)
        _cluster_copy.goal_reward_probability = np.copy(self.goal_reward_probability)
#        for g0 in range(self.n_goals):
#            for g1 in range(self.n_goals):
#                _cluster_copy.goal_rewards_received[g0, g1] = self.goal_rewards_received[g0, g1]
#                _cluster_copy.goal_reward_probability[g0, g1] = self.goal_reward_probability[g0, g1]

        return _cluster_copy



cdef class UpperDoorHypothesis(object):
    cdef int n_goals
    cdef double log_prior, alpha, goal_prior
    cdef dict cluster_assignments, clusters
    cdef list experience

    def __init__(self, int n_goals, float alpha, float goal_prior):

        self.n_goals = n_goals
        self.cluster_assignments = dict()
        self.alpha = alpha
        self.goal_prior = goal_prior

        # initialize goal clusters
        self.clusters = dict()

        # initialize posterior
        self.experience = list()
        self.log_prior = 1.0

    def update(self, int c, int seq, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef UpperDoorCluster cluster = self.clusters[k]
        cluster.update(seq, goal, r)
        self.clusters[k] = cluster

        self.experience.append((k, seq, goal, r))

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, seq, goal, r
        cdef UpperDoorCluster cluster

        #loop through experiences and get posterior
        for k, seq, goal, r in self.experience:
            cluster = self.clusters[k]
            log_likelihood += log(cluster.get_observation_probability(seq, goal, r))

        return log_likelihood
    
    def get_obs_likelihood(self, int c, int seq, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef UpperDoorCluster cluster = self.clusters[k]
        return cluster.get_observation_probability(seq, goal, r)

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.log_prior

    def get_log_prior(self):
        return self.log_prior

    def get_goal_probability(self, int c, int seq):
        cdef int g, k

        k = self.cluster_assignments[c]
        cdef UpperDoorCluster cluster = self.clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)

        cdef double [:] rew_func = cluster.get_goal_pmf(seq)
        goal_probability = np.copy(rew_func)
#        for g in range(self.n_goals):
#            goal_probability[g] = rew_func[g]
        return goal_probability


    def deep_copy(self):
        cdef UpperDoorHypothesis _h_copy = UpperDoorHypothesis(self.n_goals, self.alpha, self.goal_prior)

        cdef int k, seq, goal, r
        cdef UpperDoorCluster cluster

        _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.experience = [(k, seq, goal, r) for k, seq, goal, r in self.experience]
        _h_copy.log_prior = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = UpperDoorCluster(self.n_goals, self.goal_prior)

        self.cluster_assignments[c] = k  # note, there's no check built in here
        self.log_prior = get_prior_log_probability(self.cluster_assignments, self.alpha)




cdef class HierarchicalMappingCluster(object):

    cdef int n_sublvls, n_doors, n_primitive_actions, n_abstract_actions
    cdef double mapping_prior, alpha, roomwide_prior_log_prob
    cdef MappingCluster RoomCluster
    cdef dict roomwide_sublvl_assignments, roomwide_sublvl_clusters
    cdef list sublvl_clusters, sublvl_cluster_assignments
    cdef double[:] sublvl_wide_prior_log_prob
    cdef double tot_sublvl_wide_prior_log_prob
    cdef list sublvl_experience
    cdef double tot_sublvl_prior_log_prob
    
    def __init__(self, int n_sublvls, int n_primitive_actions, int n_abstract_actions,
             double alpha, double mapping_prior):
        
        self.n_sublvls = n_sublvls
        self.n_doors = n_sublvls+1

        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.mapping_prior = mapping_prior
        self.alpha = alpha

        self.RoomCluster = MappingCluster(self.n_primitive_actions, 
                                           self.n_abstract_actions, 
                                           self.mapping_prior)
        
        # room-wide CRP for sublvl clusters
        self.roomwide_sublvl_assignments = dict()
        self.roomwide_sublvl_clusters = dict()

        # store the prior probability
        self.roomwide_prior_log_prob = 1.0/2

        # sublvl CRP for sublvl clusters
        self.sublvl_clusters = [dict() for ii in range(n_sublvls)]
        self.sublvl_cluster_assignments = []
        
        self.sublvl_wide_prior_log_prob = np.ones(n_sublvls)*1.0/n_sublvls/2
        self.tot_sublvl_wide_prior_log_prob = 1.0/2

        # need to store all experiences for log likelihood calculations
        self.sublvl_experience = list()
        
        self.tot_sublvl_prior_log_prob = 1.0
        
        
    def add_new_sublvl_context_assignment(self, int c, list hierarchical_assignment, 
                                          MappingCluster new_sublvl_cluster):
        cdef int k
        cdef MappingCluster sublvl_cluster = None
        
        assert len(hierarchical_assignment) == 2
        
        # check if new context has assignment in room-wide CRP and make updates if needed
        cdef dict room_assignments = hierarchical_assignment[0]
        if c in room_assignments.keys():
            k = room_assignments[c]
            
            # check if cluster "k" is already been assigned new cluster
            if k not in self.roomwide_sublvl_assignments.values():
                assert new_sublvl_cluster is not None
                # if not, add a new mapping cluster, which should have been passed in 
                # from environment-wide CRP
                self.roomwide_sublvl_clusters[k] = new_sublvl_cluster

            self.roomwide_sublvl_assignments[c] = k
            self.roomwide_prior_log_prob = get_prior_log_probability(
                    self.roomwide_sublvl_assignments, self.alpha)
            
            self.tot_sublvl_prior_log_prob = self.roomwide_prior_log_prob + self.tot_sublvl_wide_prior_log_prob
            
            sublvl_cluster = self.roomwide_sublvl_clusters[k]

        # update sublvl CRP
        cdef list sublvl_assignments = hierarchical_assignment[-1]

        cdef list sublvl_list = [ ii for ii, _subassignment in enumerate(sublvl_assignments) if c in _subassignment.keys()]
        assert len(sublvl_list) == 1

        cdef int sublvl = sublvl_list[0]
        k = sublvl_assignments[sublvl][c]
        
        # expand list of sublvl_cluster_assignments if necessary
        cdef int n_sublvl_CRPs = len(self.sublvl_cluster_assignments)
        if sublvl >= n_sublvl_CRPs:
            assert sublvl == n_sublvl_CRPs and n_sublvl_CRPs < self.n_sublvls
               # first condition ensures sublvl indexes element
               # that will be appended to self.sublvl_cluster_assignments
            self.sublvl_cluster_assignments.append(dict())
        
        # note: these are shallow copies (ie. aliases)
        cdef dict subassignments = self.sublvl_cluster_assignments[sublvl]
        cdef dict sub_clusters = self.sublvl_clusters[sublvl]
        
        # check if cluster "k" is already been assigned new cluster
        if k not in subassignments.values():
            assert sublvl_cluster is not None
            # if not, add a new mapping cluster
            sub_clusters[k] = sublvl_cluster

        subassignments[c] = k  # note, there's no check built in here
        self.sublvl_wide_prior_log_prob[sublvl] = get_prior_log_probability(subassignments, self.alpha)
        
        self.tot_sublvl_wide_prior_log_prob = np.sum(self.sublvl_wide_prior_log_prob)
        self.tot_sublvl_prior_log_prob = self.roomwide_prior_log_prob + self.tot_sublvl_wide_prior_log_prob
        
    def update_mapping(self, int c, int a, int aa):
        cdef list sublvl_list 
        cdef int sublvl, k
        cdef MappingCluster cluster
        
        # check whether we should update room mapping or sublvl mapping
        if c % self.n_doors == 0:
            # update room mapping
            self.RoomCluster.update(a, aa)
        else:
            # update sublvl mapping
            sublvl_list = [ ii for ii, _subassignment in enumerate(self.sublvl_cluster_assignments) if c in _subassignment.keys()]
            assert len(sublvl_list) == 1
            sublvl = sublvl_list[0]
            k = self.sublvl_cluster_assignments[sublvl][c]

            cluster = self.sublvl_clusters[sublvl][k]
            cluster.update(a, aa)

            # need to store all experiences for log likelihood calculations
            self.sublvl_experience.append((sublvl, k, a, aa))
            
    def get_mapping_probability(self, int c, int a, int aa):
        cdef list sublvl_list 
        cdef int sublvl, k
        cdef MappingCluster cluster

        # check whether we want room mapping or sublvl mapping
        if c % self.n_doors == 0:
            # return room mapping
            return self.RoomCluster.get_mapping_mle(a, aa)
        else:
            sublvl_list = [ ii for ii, _subassignment in enumerate(self.sublvl_cluster_assignments) if c in _subassignment.keys()]
            assert len(sublvl_list) == 1
            sublvl = sublvl_list[0]
            k = self.sublvl_cluster_assignments[sublvl][c]
        
            cluster = self.sublvl_clusters[sublvl][k]
            return cluster.get_mapping_mle(a, aa)
        
    def get_sublvl_assignments(self):
        return [self.roomwide_sublvl_assignments, self.sublvl_cluster_assignments]
    
    def get_log_prior(self):
        return self.tot_sublvl_prior_log_prob
    
    def get_RoomCluster(self):
        return self.RoomCluster
    
    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int sublvl, k, a, aa
        cdef MappingCluster cluster

        #loop through experiences and get posterior
        for sublvl, k, a, aa in self.sublvl_experience:
            cluster = self.sublvl_clusters[sublvl][k]
            log_likelihood += log(cluster.get_likelihood(a, aa))

        return log_likelihood

    def get_obs_likelihood(self, int c, int a, int aa):
        cdef list sublvl_list 
        cdef int sublvl, k
        cdef MappingCluster cluster

        # check whether we want room mapping or sublvl mapping
        if c % self.n_doors == 0:
            # return room mapping
            return self.RoomCluster.get_likelihood(a, aa)
        else:
            sublvl_list = [ ii for ii, _subassignment in enumerate(self.sublvl_cluster_assignments) if c in _subassignment.keys()]
            assert len(sublvl_list) == 1
            sublvl = sublvl_list[0]
            k = self.sublvl_cluster_assignments[sublvl][c]
        
            cluster = self.sublvl_clusters[sublvl][k]
            return cluster.get_likelihood(a, aa)

    def deep_copy(self, dict old_sublvl_clusters, dict new_sublvl_clusters):
        cdef HierarchicalMappingCluster _h_copy = HierarchicalMappingCluster(
            self.n_sublvls, self.n_primitive_actions, self.n_abstract_actions, 
            self.alpha, self.mapping_prior
        )
        
        cdef int k, k2, ii, a, aa, sublvl
        cdef MappingCluster cluster

        _h_copy.RoomCluster = self.RoomCluster.deep_copy()
        
        # room-wide CRP for sublvl clusters
        _h_copy.roomwide_sublvl_assignments = {c: k for c, k in self.roomwide_sublvl_assignments.iteritems()}
        
        _h_copy.roomwide_sublvl_clusters = dict()
        for k, cluster in self.roomwide_sublvl_clusters.iteritems():
            for k2, cluster_old in old_sublvl_clusters.iteritems():
                if cluster == cluster_old:
                    _h_copy.roomwide_sublvl_clusters[k] = new_sublvl_clusters[k2]
                    break

        # store the prior probability
        _h_copy.roomwide_prior_log_prob = self.roomwide_prior_log_prob

        # sublvl CRP for sublvl clusters
        _h_copy.sublvl_clusters = [ {k: cluster.deep_copy() for k, cluster in sublvl_clustering.iteritems()} 
            for sublvl_clustering in self.sublvl_clusters]
        _h_copy.sublvl_cluster_assignments = [ {c:k for c, k in assignments.iteritems()}
            for assignments in self.sublvl_cluster_assignments]
        
        _h_copy.sublvl_clusters = [dict() for ii in range(self.n_sublvls)]
        for ii in range(self.n_sublvls):
            for k, cluster in self.sublvl_clusters[ii].iteritems():
                for k2, cluster_old in self.roomwide_sublvl_clusters.iteritems():
                    if cluster == cluster_old:
                        _h_copy.sublvl_clusters[ii][k] = _h_copy.roomwide_sublvl_clusters[k2]
                        break
        
        _h_copy.sublvl_wide_prior_log_prob = np.copy(self.sublvl_wide_prior_log_prob)
        _h_copy.tot_sublvl_wide_prior_log_prob = self.tot_sublvl_wide_prior_log_prob

        _h_copy.sublvl_experience = [(sublvl, k, a, aa) for sublvl, k, a, aa in self.sublvl_experience]

        _h_copy.tot_sublvl_prior_log_prob = self.tot_sublvl_prior_log_prob

        return _h_copy
        

cdef class HierarchicalMappingHypothesis(object):

    cdef int n_sublvls, n_doors, n_primitive_actions, n_abstract_actions
    cdef double alpha, mapping_prior
    cdef dict room_cluster_assignments, room_clusters
    cdef list room_experience
    cdef double room_prior_log_prob, sublvl_prior_log_prob, 
    cdef dict sublvl_assignments, sublvl_clusters
    cdef double sublvl_CRP_prior_log_prob, prior_log_prob
    
    # note: sublvl clusters within a hypothesis will be modelled via a 
    # hierarchical Dirichlet process to allow sharing of clusters across sublvls
    # and upper levels

    def __init__(self, int n_sublvls, int n_primitive_actions, int n_abstract_actions,
                 double alpha, double mapping_prior):

        self.n_sublvls = n_sublvls
        self.n_doors = n_sublvls+1
        
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.alpha = alpha
        self.mapping_prior = mapping_prior

        # initialize the room clusters
        self.room_cluster_assignments = dict()
        self.room_clusters = dict()
        self.room_experience = list()
        
        # store the prior probability
        self.room_prior_log_prob = 1./3

        # environment-wide CRP for sublvl clusters
        self.sublvl_assignments = dict()
        self.sublvl_clusters = dict()

        # store the prior probability
        self.sublvl_prior_log_prob = 1./3
        
        self.sublvl_CRP_prior_log_prob = 1./3
        
        self.prior_log_prob = 1.
       
        
    def add_new_room_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.room_cluster_assignments.values():
            # if not, add a new mapping cluster
            self.room_clusters[k] = HierarchicalMappingCluster(self.n_sublvls,
                              self.n_primitive_actions, self.n_abstract_actions,
                              self.alpha, self.mapping_prior)

        self.room_cluster_assignments[c] = k
        self.room_prior_log_prob = get_prior_log_probability(self.room_cluster_assignments, self.alpha)
        
        self.prior_log_prob = self.room_prior_log_prob + self.sublvl_prior_log_prob + self.sublvl_CRP_prior_log_prob


    def add_new_sublvl_context_assignment(self, int c, list hierarchical_assignment):
        cdef int k, room_c, room_k
        cdef MappingCluster sublvl_cluster = None
        cdef HierarchicalMappingCluster RoomCluster
        
        # check if new context has assignment in environment-wide CRP
        cdef dict env_assignments = hierarchical_assignment[0]
        if c in env_assignments.keys():
            k = env_assignments[c]
            
            # check if cluster "k" has already been assigned new cluster
            if k not in self.sublvl_clusters.keys():
                # if not, add a new mapping cluster
                self.sublvl_clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)
                
            self.sublvl_assignments[c] = k
            self.sublvl_prior_log_prob = get_prior_log_probability(self.sublvl_assignments, self.alpha)
            sublvl_cluster = self.sublvl_clusters[k]
            
        room_c = c - (c % self.n_doors)
        room_k = self.room_cluster_assignments[room_c]
        RoomCluster = self.room_clusters[room_k]
        
        RoomCluster.add_new_sublvl_context_assignment(c, hierarchical_assignment[1:], sublvl_cluster)
        
        self.sublvl_CRP_prior_log_prob = sum([cluster.get_log_prior() for cluster in self.room_clusters.values()])
        
        self.prior_log_prob = self.room_prior_log_prob + self.sublvl_prior_log_prob + self.sublvl_CRP_prior_log_prob
        

    def update_mapping(self, int c, int a, int aa):
        # get room cluster to update
        cdef int room_c = c - (c % self.n_doors)
        cdef int room_k = self.room_cluster_assignments[room_c]
        cdef HierarchicalMappingCluster RoomCluster = self.room_clusters[room_k]

        RoomCluster.update_mapping(c, a, aa)

        # if updating mapping for a room, need to store all experiences 
        # for log likelihood calculations
        if c % self.n_doors == 0:
            self.room_experience.append((room_k, a, aa))


    def get_mapping_probability(self, int c, int a, int aa):
        # get room cluster to update
        cdef int room_c = c - (c % self.n_doors)
        cdef int room_k = self.room_cluster_assignments[room_c]
        cdef HierarchicalMappingCluster RoomCluster = self.room_clusters[room_k]
        
        return RoomCluster.get_mapping_probability(c, a, aa)
        

    def get_room_assignments(self):
        return self.room_cluster_assignments


    def get_sublvl_assignments(self, int c):
        # returns hierarchy of sublvl assignments that includes context c
        cdef list hierarchical_assignment = [self.sublvl_assignments]
        
        cdef int room_c = c - (c % self.n_doors)
        cdef int room_k = self.room_cluster_assignments[room_c]
        cdef HierarchicalMappingCluster RoomCluster = self.room_clusters[room_k]
        
        hierarchical_assignment += RoomCluster.get_sublvl_assignments()
        
        return hierarchical_assignment

    def get_log_prior(self):
        return self.prior_log_prob
    
    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, a, aa
        cdef HierarchicalMappingCluster RoomCluster
        cdef MappingCluster MapCluster

        #loop through experiences and get posterior
        for k, a, aa in self.room_experience:
            RoomCluster = self.room_clusters[k]
            MapCluster = RoomCluster.get_RoomCluster()
            log_likelihood += log(MapCluster.get_likelihood(a, aa))
            
        for RoomCluster in self.room_clusters.values():
            log_likelihood += RoomCluster.get_log_likelihood()

        return log_likelihood

    def get_obs_likelihood(self, int c, int a, int aa):
        # get room cluster for context
        cdef int room_c = c - (c % self.n_doors)
        cdef int room_k = self.room_cluster_assignments[room_c]
        cdef HierarchicalMappingCluster RoomCluster = self.room_clusters[room_k]
        
        return RoomCluster.get_obs_likelihood(c, a, aa)

    def get_log_posterior(self):
        return self.prior_log_prob + self.get_log_likelihood()

    def deep_copy(self):
        cdef HierarchicalMappingHypothesis _h_copy = HierarchicalMappingHypothesis(
            self.n_sublvls, self.n_primitive_actions, self.n_abstract_actions, 
            self.alpha, self.mapping_prior
        )

        cdef int k, a, aa
        cdef HierarchicalMappingCluster RoomCluster
        cdef MappingCluster cluster
        
        
        # environment-wide CRP for sublvl clusters
        _h_copy.sublvl_assignments = {c:k for c, k in self.sublvl_assignments.iteritems()}
        _h_copy.sublvl_clusters = {k: cluster.deep_copy() for k, cluster in self.sublvl_clusters.iteritems()}

        # initialize the room clusters
        old_sublvl_clusters = self.sublvl_clusters
        new_sublvl_clusters = _h_copy.sublvl_clusters
        _h_copy.room_clusters = {k: RoomCluster.deep_copy(old_sublvl_clusters, new_sublvl_clusters) for k, RoomCluster in self.room_clusters.iteritems()}
        _h_copy.room_cluster_assignments = {c:k for c,k in self.room_cluster_assignments.iteritems()}
        _h_copy.room_experience = [(k,a,aa) for k,a,aa in self.room_experience]
        
        # store the prior probability
        _h_copy.room_prior_log_prob = self.room_prior_log_prob

        # store the prior probability
        _h_copy.sublvl_prior_log_prob = self.sublvl_prior_log_prob
        
        _h_copy.sublvl_CRP_prior_log_prob = self.sublvl_CRP_prior_log_prob
        
        _h_copy.sublvl_CRP_prior_log_prob = self.prior_log_prob
        
        return _h_copy



