# cython: profile=True, linetrace=True, boundscheck=True, wraparound=True
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from core import get_prior_log_probability
from math import isnan, isinf

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
    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a, log_likelihoods
    cdef double [:] abstract_action_counts, primitive_action_counts
    cdef int n_primitive_actions, n_abstract_actions
    cdef double mapping_prior

    def __cinit__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):

        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a, log_likelihoods
        cdef double[:] abstract_action_counts, primitive_action_counts

        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=float) * mapping_prior
        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=float) *  mapping_prior * n_primitive_actions
        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=float) * \
                      (1.0 / n_primitive_actions)

        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE) * mapping_prior * n_abstract_actions
        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE) * \
                        (1.0 / n_abstract_actions)
        log_likelihoods = np.zeros((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE)


        self.mapping_history = mapping_history
        self.abstract_action_counts = abstract_action_counts
        self.mapping_mle = mapping_mle
        self.primitive_action_counts = primitive_action_counts
        self.pr_aa_given_a = pr_aa_given_a
        self.log_likelihoods = log_likelihoods

        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.mapping_prior = mapping_prior

    cdef update(self, int a, int aa):
        cdef int aa0, a0
        cdef double n

        self.mapping_history[a, aa] += 1.0
        self.abstract_action_counts[aa] += 1.0
        self.primitive_action_counts[a] += 1.0
        
        for aa0 in range(self.n_abstract_actions + 1):
            # p(A|a, k) estimator
            self.pr_aa_given_a[a, aa0] = self.mapping_history[a, aa0] / self.primitive_action_counts[a]
            
            n = self.mapping_history[a, aa0] - self.mapping_prior
            self.log_likelihoods[a, aa0] = n*log(self.pr_aa_given_a[a, aa0])

        for a0 in range(self.n_primitive_actions):
            self.mapping_mle[a0, aa] = self.mapping_history[a0, aa] / self.abstract_action_counts[aa]

    cdef get_mapping_mle(self, int a, int aa):
        return self.mapping_mle[a, aa]

    cdef get_likelihood(self, int a, int aa):
        return self.pr_aa_given_a[a, aa]

    cdef get_log_likelihood(self):
        cdef double log_likelihood = np.sum(self.log_likelihoods)
        assert not isnan(log_likelihood) and not isinf(log_likelihood)
        return log_likelihood

    def deep_copy(self):
        cdef MappingCluster _cluster_copy = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                                           self.mapping_prior)
        
        _cluster_copy.primitive_action_counts = np.copy(self.primitive_action_counts)
        _cluster_copy.mapping_history = np.copy(self.mapping_history)
        _cluster_copy.mapping_mle = np.copy(self.mapping_mle)
        _cluster_copy.pr_aa_given_a = np.copy(self.pr_aa_given_a)
        _cluster_copy.log_likelihoods = np.array(self.log_likelihoods)
        _cluster_copy.abstract_action_counts = np.copy(self.abstract_action_counts)

        return _cluster_copy


cdef class MappingHypothesis(object):

    cdef dict cluster_assignments, clusters
    cdef double prior_log_prob, alpha, mapping_prior
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

    cpdef update_mapping(self, int c, int a, int aa):
        cdef int k = self.cluster_assignments[c]
        cdef MappingCluster cluster = self.clusters[k]
        cluster.update(a, aa)
        self.clusters[k] = cluster

    cpdef double get_log_likelihood(self):
        cdef MappingCluster cluster
        cdef double log_likelihood = 0

        #loop through clusters and get log_likelihood of data stored there
        for cluster in self.clusters.values():
            log_likelihood += cluster.get_log_likelihood()
        
        return log_likelihood

    cpdef double get_log_posterior(self):
        return self.prior_log_prob + self.get_log_likelihood()

    cpdef get_mapping_probability(self, int c, int a, int aa):
        cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
        return cluster.get_mapping_mle(a, aa)

    def get_log_prior(self):
        return self.prior_log_prob

    def deep_copy(self):
        cdef MappingHypothesis _h_copy = MappingHypothesis(
            self.n_primitive_actions, self.n_abstract_actions, self.alpha,
            self.mapping_prior
        )

        cdef int k
        cdef MappingCluster cluster

        _h_copy.cluster_assignments = dict(self.cluster_assignments)
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.prior_log_prob = self.prior_log_prob

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)

        self.cluster_assignments[c] = k
        self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)


cdef class GoalCluster(object):
    cdef int n_goals
    cdef double set_visits, goal_prior
    cdef double [:]  goal_rewards_received, goal_reward_probability
    cdef double [:,::1] goal_visits

    def __cinit__(self, int n_goals, float goal_prior):
        self.n_goals = n_goals
        self.goal_prior = goal_prior

        # rewards!
        self.set_visits =  n_goals * goal_prior
        self.goal_rewards_received = np.ones(n_goals) * goal_prior
        self.goal_reward_probability = np.ones(n_goals) * (1.0 / n_goals)
        self.goal_visits = np.zeros((n_goals,2))

    cdef update(self, int goal, int r):
        cdef double r0
        cdef int g0

        self.set_visits += 1.0
        self.goal_rewards_received[goal] += r

        if r == 0:
            self.goal_visits[goal,0] += 1
            r0 = 1.0 / (self.n_goals - 1)
            for g0 in range(self.n_goals):
                if g0 != goal:
                    self.goal_rewards_received[g0] += r0
        else:
            self.goal_visits[goal,1] += 1

        # update all goal probabilities
        for g0 in range(self.n_goals):
            self.goal_reward_probability[g0] = self.goal_rewards_received[g0] / self.set_visits

    cpdef double get_observation_probability(self, int goal, int r):
        if r == 0:
            return 1 - self.goal_reward_probability[goal]
        return self.goal_reward_probability[goal]

    def get_goal_pmf(self):
        return self.goal_reward_probability

    cdef get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int g
        for g in range(self.n_goals):
            log_likelihood += self.goal_visits[g,0]*log(self.get_observation_probability(g, 0)) + self.goal_visits[g,1]*log(self.get_observation_probability(g, 1))

        return log_likelihood
    
    cpdef deep_copy(self):
        cdef GoalCluster _cluster_copy = GoalCluster(self.n_goals, self.goal_prior)

        _cluster_copy.set_visits = self.set_visits
        _cluster_copy.goal_rewards_received = np.copy(self.goal_rewards_received)
        _cluster_copy.goal_reward_probability = np.copy(self.goal_reward_probability)
        _cluster_copy.goal_visits = np.array(self.goal_visits)

        return _cluster_copy



cdef class GoalHypothesis(object):
    cdef int n_goals
    cdef double log_prior, alpha, goal_prior
    cdef dict cluster_assignments, clusters

    def __init__(self, int n_goals, float alpha, float goal_prior):

        self.n_goals = n_goals
        self.cluster_assignments = dict()
        self.alpha = alpha
        self.goal_prior = goal_prior

        # initialize goal clusters
        self.clusters = dict()

        # initialize posterior
        self.log_prior = 1.0

    cpdef update(self, int c, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        cluster.update(goal, r)
        self.clusters[k] = cluster

    cpdef get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef GoalCluster cluster

        for cluster in self.clusters.values():
            log_likelihood += cluster.get_log_likelihood()
        
        return log_likelihood

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
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability


    def deep_copy(self):
        cdef GoalHypothesis _h_copy = GoalHypothesis(self.n_goals, self.alpha, self.goal_prior)

        cdef int k
        cdef GoalCluster cluster

        _h_copy.cluster_assignments = dict(self.cluster_assignments)
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.log_prior = self.log_prior

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


cdef class HierarchicalHypothesis(object):
    cdef int n_goals, n_primitive_actions, n_abstract_actions
    cdef double gamma, alpha_r, alpha_hi, alpha_lo
    cdef dict room_assignments, room_clusters, goal_assignments, goal_clusters, mapping_assignments, mapping_clusters
    cdef double room_log_prior, goal_log_prior, mapping_log_prior, goal_prior, mapping_prior


    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions, 
                 float gamma, double alpha_r, double alpha_hi, double alpha_lo, 
                 float goal_prior, float mapping_prior):

        self.n_goals = n_goals
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.gamma = gamma
        self.alpha_r = alpha_r
        self.alpha_hi = alpha_hi
        self.alpha_lo = alpha_lo
        
        self.room_assignments = dict()
        self.goal_assignments = dict()
        self.mapping_assignments = dict()
        
        self.room_clusters = dict()
        self.goal_clusters = dict()
        self.mapping_clusters = dict()

        # initialize posterior
        self.room_log_prior = 0.
        self.goal_log_prior = 0.
        self.mapping_log_prior = 0.

        self.goal_prior = goal_prior
        self.mapping_prior = mapping_prior
    
    # goal update
    cpdef update(self, int c, int goal, int r):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        cluster.update(c, goal, r)
        
    cpdef update_mapping(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        cluster.update_mapping(c, a, aa)
        
    cpdef get_goal_probability(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.get_goal_probability(c)

    cpdef get_mapping_probability(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.get_mapping_probability(c, a, aa)

    cpdef get_log_posterior(self):
        cdef HierarchicalRoomCluster RoomCluster
        cdef MappingCluster m_cluster
        cdef GoalCluster g_cluster
        cdef double log_posterior = self.room_log_prior + self.mapping_log_prior + self.goal_log_prior
        log_posterior += sum([RoomCluster.goal_log_prior() + RoomCluster.mapping_log_prior() for RoomCluster in self.room_clusters.values()])
        log_posterior += sum([m_cluster.get_log_likelihood() for m_cluster in self.mapping_clusters.values()])
        log_posterior += sum([g_cluster.get_log_likelihood() for g_cluster in self.goal_clusters.values()])
        return log_posterior
    
    cpdef get_room_assignments(self):
        return self.room_assignments
    
    cpdef get_goal_assignments(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]

        cdef list goal_assignments = [self.goal_assignments, cluster.get_goal_assignments()]
        
        return goal_assignments

    cpdef get_mapping_assignments(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]

        cdef list mapping_assignments = [self.mapping_assignments, cluster.get_mapping_assignments()]
        
        return mapping_assignments

    cpdef add_new_room_context_assignment(self, int c, int k):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.room_clusters.keys():
            # if not, add an new goal cluster
            self.room_clusters[k] = HierarchicalRoomCluster(self.n_goals, self.n_primitive_actions, 
                              self.n_abstract_actions, self.gamma, self.alpha_lo, 
                              self.goal_prior, self.mapping_prior)

        self.room_assignments[c] = k  # note, there's no check built in here
        self.room_log_prior = get_prior_log_probability(self.room_assignments, self.alpha_r)
        
    cpdef add_new_goal_context_assignment(self, int c, list hierarchical_assignment):
        cdef int k
        cdef GoalCluster cluster = None
        
        # check if new context has assignment in environment-wide CRP
        cdef dict env_assignments = hierarchical_assignment[0]
        if c in env_assignments.keys():
            k = env_assignments[c]
            
            # check if cluster "k" has already been assigned new cluster
            if k not in self.goal_clusters.keys():
                # if not, add an new goal cluster
                self.goal_clusters[k] = GoalCluster(self.n_goals, self.goal_prior)

            self.goal_assignments[c] = k  # note, there's no check built in here
            self.goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha_hi)
            cluster = self.goal_clusters[k]
            
        k = self.room_assignments[c]
        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
        assert RoomCluster is not None
        
        k = hierarchical_assignment[1][c]
        RoomCluster.add_new_goal_context_assignment(c, k, cluster)
        
    cpdef add_new_mapping_context_assignment(self, int c, list hierarchical_assignment):
        cdef int k
        cdef MappingCluster cluster = None
        
        # check if new context has assignment in environment-wide CRP
        cdef dict env_assignments = hierarchical_assignment[0]
        if c in env_assignments.keys():
            k = env_assignments[c]
            
            # check if cluster "k" has already been assigned new cluster
            if k not in self.mapping_clusters.keys():
                # if not, add an new goal cluster
                self.mapping_clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                                  self.mapping_prior)
                
            self.mapping_assignments[c] = k  # note, there's no check built in here
            self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha_hi)
            cluster = self.mapping_clusters[k]
            
        k = self.room_assignments[c]
        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
        assert RoomCluster is not None
        
        k = hierarchical_assignment[1][c]
        RoomCluster.add_new_mapping_context_assignment(c, k, cluster)

    cpdef deep_copy(self):
        cdef MappingCluster m_cluster
        cdef GoalCluster g_cluster
        cdef HierarchicalRoomCluster room_cluster
        cdef int k
        
        cdef HierarchicalHypothesis _h_copy = HierarchicalHypothesis(self.n_goals, 
                self.n_primitive_actions, self.n_abstract_actions, 
                self.gamma, self.alpha_r, self.alpha_hi, self.alpha_lo, self.goal_prior, self.mapping_prior)

        _h_copy.room_log_prior = self.room_log_prior
        _h_copy.goal_log_prior = self.goal_log_prior
        _h_copy.mapping_log_prior = self.mapping_log_prior
        
        _h_copy.room_assignments = dict(self.room_assignments)
        _h_copy.goal_assignments = dict(self.goal_assignments)
        _h_copy.mapping_assignments = dict(self.mapping_assignments)
        
        _h_copy.goal_clusters = {k:g_cluster.deep_copy() for k,g_cluster in self.goal_clusters.iteritems()}
        _h_copy.mapping_clusters = {k:m_cluster.deep_copy() for k,m_cluster in self.mapping_clusters.iteritems()}
        
        _h_copy.room_clusters = {k:room_cluster.deep_copy(self.goal_clusters, _h_copy.goal_clusters, self.mapping_clusters, _h_copy.mapping_clusters) for k,room_cluster in self.room_clusters.iteritems()}

        return _h_copy


cdef class HierarchicalRoomCluster(object):
    cdef int n_goals, n_primitive_actions, n_abstract_actions
    cdef double gamma, alpha
    cdef dict room_assignments, room_clusters, _goal_assignments, goal_clusters, _mapping_assignments, mapping_clusters
    cdef double _goal_log_prior, _mapping_log_prior, goal_prior, mapping_prior

    def __cinit__(self, int n_goals, int n_primitive_actions, int n_abstract_actions, 
                 float gamma, float alpha,
                 float goal_prior, float mapping_prior):
        
        self.n_goals = n_goals
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.gamma = gamma
        self.alpha = alpha
        
        self._goal_assignments = dict()
        self._mapping_assignments = dict()
        
        self.goal_clusters = dict()
        self.mapping_clusters = dict()

        # initialize posterior
        self._goal_log_prior = 0.
        self._mapping_log_prior = 0.

        self.goal_prior = goal_prior
        self.mapping_prior = mapping_prior
        
    # goal update
    cdef update(self, int c, int goal, int r):
        cdef int k = self._goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]
        cluster.update(goal, r)
        
    cdef update_mapping(self, int c, int a, int aa):
        cdef int k = self._mapping_assignments[c]
        cdef MappingCluster cluster = self.mapping_clusters[k]
        cluster.update(a, aa)
    
    cdef get_goal_probability(self, int c):
        cdef int k = self._goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)
        cdef double [:] rew_func = cluster.goal_reward_probability

        cdef int g
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability

    cdef get_mapping_probability(self, int c, int a, int aa):
        cdef MappingCluster cluster = self.mapping_clusters[self._mapping_assignments[c]]
        return cluster.mapping_mle[a, aa]

    cdef get_goal_assignments(self):
        return self._goal_assignments
    
    cdef get_mapping_assignments(self):
        return self._mapping_assignments

    cdef add_new_goal_context_assignment(self, int c, int k, GoalCluster cluster):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.goal_clusters.keys():
            # if not, add an new goal cluster
            assert cluster is not None
            self.goal_clusters[k] = cluster

        self._goal_assignments[c] = k  # note, there's no check built in here
        self._goal_log_prior = get_prior_log_probability(self._goal_assignments, self.alpha)

    cdef add_new_mapping_context_assignment(self, int c, int k, MappingCluster cluster):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.mapping_clusters.keys():
            # if not, add an new goal cluster
            assert cluster is not None
            self.mapping_clusters[k] = cluster

        self._mapping_assignments[c] = k  # note, there's no check built in here
        self._mapping_log_prior = get_prior_log_probability(self._mapping_assignments, self.alpha)
        
    cdef goal_log_prior(self):
        return self._goal_log_prior

    cdef mapping_log_prior(self):
        return self._mapping_log_prior
    
    cdef deep_copy(self, dict old_goal_clusters, dict new_goal_clusters, dict old_mapping_clusters, dict new_mapping_clusters):
        cdef int k, k2
        cdef GoalCluster goal_cluster, goal_cluster_old
        cdef MappingCluster mapping_cluster, mapping_cluster_old
        
        _cluster_copy = HierarchicalRoomCluster(self.n_goals, self.n_primitive_actions, 
                                                self.n_abstract_actions, self.gamma, self.alpha, 
                                                self.goal_prior, self.mapping_prior)

        _cluster_copy._goal_log_prior = self._goal_log_prior 
        _cluster_copy._mapping_log_prior = self._mapping_log_prior

        _cluster_copy._goal_assignments = dict(self._goal_assignments)
        _cluster_copy._mapping_assignments = dict(self._mapping_assignments)
        
        _cluster_copy.goal_clusters = dict()
        for k, goal_cluster in self.goal_clusters.iteritems():
            for k2, goal_cluster_old in old_goal_clusters.iteritems():
                if goal_cluster == goal_cluster_old:
                    _cluster_copy.goal_clusters[k] = new_goal_clusters[k2]
                    break

        _cluster_copy.mapping_clusters = dict()
        for k, mapping_cluster in self.mapping_clusters.iteritems():
            for k2, mapping_cluster_old in old_mapping_clusters.iteritems():
                if mapping_cluster == mapping_cluster_old:
                    _cluster_copy.mapping_clusters[k] = new_mapping_clusters[k2]
                    break

        return _cluster_copy