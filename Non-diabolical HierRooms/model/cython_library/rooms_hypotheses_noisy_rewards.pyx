# cython: profile=True, linetrace=True, boundscheck=True, wraparound=True
from __future__ import division
import random
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

    def __init__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):

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

        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.mapping_prior = mapping_prior
        
        self.log_likelihoods = log_likelihoods

    def update(self, int a, int aa):
        cdef int aa0, a0
        cdef double n
        self.mapping_history[a, aa] += 1.0
        self.abstract_action_counts[aa] += 1.0
        self.primitive_action_counts[a] += 1.0

        for aa0 in range(self.n_abstract_actions + 1):
            # self.mapping_mle[a, aa0] = self.mapping_history[a, aa0] / self.abstract_action_counts[aa0]

            # p(A|a, k) estimator
            self.pr_aa_given_a[a, aa0] = self.mapping_history[a, aa0] / self.primitive_action_counts[a]
            
            n = self.mapping_history[a, aa0] - self.mapping_prior
            self.log_likelihoods[a, aa0] = n*log(self.pr_aa_given_a[a, aa0])

        for a0 in range(self.n_primitive_actions):
            self.mapping_mle[a0, aa] = self.mapping_history[a0, aa] / self.abstract_action_counts[aa]

            # p(A|a, k) estimator
            # self.pr_aa_given_a[a0, aa] = self.mapping_history[a0, aa] / self.primitive_action_counts[a0]
            
            # n = self.mapping_history[a0, aa] - self.mapping_prior
            # self.log_likelihoods[a0, aa] = n*log(self.pr_aa_given_a[a0, aa])        

    def get_mapping_mle(self, int a, int aa):
        return self.mapping_mle[a, aa]

    def get_likelihood(self, int a, int aa):
        return self.pr_aa_given_a[a, aa]

    def log_likelihood(self):
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
        _cluster_copy.abstract_action_counts = np.copy(self.abstract_action_counts)
        _cluster_copy.log_likelihoods = np.array(self.log_likelihoods)

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

    def update_mapping(self, int c, int a, int aa):
        cdef int k = self.cluster_assignments[c]
        cdef MappingCluster cluster = self.clusters[k]
        cluster.update(a, aa)
        self.clusters[k] = cluster

    def get_log_likelihood(self):
        cdef MappingCluster cluster
        cdef double log_likelihood = 0

        #loop through clusters and get log_likelihood of data stored there
        for cluster in self.clusters.values():
            log_likelihood += cluster.log_likelihood()
        
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
            self.mapping_prior)

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
            # if not, add a new mapping cluster
            self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)

        self.cluster_assignments[c] = k
        self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)


cdef class GoalCluster(object):
    cdef double set_visits, goal_prior
    cdef double [:]  goal_rewards_received, goal_reward_probability
    cdef double [:,::1] goal_visits
    cdef int n_goals

    def __init__(self, int n_goals, float goal_prior):
        self.n_goals = n_goals
        self.goal_prior = goal_prior

        # rewards!
        self.set_visits =  n_goals * goal_prior
        self.goal_rewards_received = np.ones(n_goals) * goal_prior
        self.goal_reward_probability = np.ones(n_goals) * (1.0 / n_goals)
        self.goal_visits = np.zeros((n_goals,2))

    def update(self, int goal, int r):
        cdef double r0
        cdef int g0, rand_sample_size

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

    def get_observation_probability(self, int goal, int r):
        if r == 0:
            return 1 - self.goal_reward_probability[goal]
        return self.goal_reward_probability[goal]
    
    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int g
        for g in range(self.n_goals):
            log_likelihood += self.goal_visits[g,0]*log(self.get_observation_probability(g, 0)) + self.goal_visits[g,1]*log(self.get_observation_probability(g, 1))

        return log_likelihood

    def get_goal_pmf(self):
        return self.goal_reward_probability

    def deep_copy(self):
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
        self.alpha = alpha
        self.goal_prior = goal_prior

        # initialize goal clusters
        self.cluster_assignments = dict()
        self.clusters = dict()

        # initialize posterior
        self.log_prior = 1.0

    def update(self, int c, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        cluster.update(goal, r)
        self.clusters[k] = cluster

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef GoalCluster cluster

        for cluster in self.clusters.values():
            log_likelihood += cluster.get_log_likelihood()
        
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

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = GoalCluster(self.n_goals, self.goal_prior)

        self.cluster_assignments[c] = k  # note, there's no check built in here
        self.log_prior = get_prior_log_probability(self.cluster_assignments, self.alpha)

    def deep_copy(self):
        cdef GoalHypothesis _h_copy = GoalHypothesis(self.n_goals, self.alpha, self.goal_prior)

        cdef int k
        cdef GoalCluster cluster

        _h_copy.cluster_assignments = dict(self.cluster_assignments)
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.log_prior = self.log_prior

        return _h_copy


cdef class DoorSeqCluster(object):
    cdef int n_goals, n_sublvls
    cdef double alpha
    cdef double [:] _door_log_prior
    cdef list door_assignments, door_clusters

    def __init__(self, int n_goals, double alpha):
        cdef int ii
        
        self.n_goals = n_goals
        self.n_sublvls = n_goals-1

        self.door_assignments = [{} for ii in range(self.n_goals)]
        self.door_clusters = [{} for ii in range(self.n_goals)]
        self._door_log_prior = np.zeros(self.n_goals, dtype=DTYPE)
        self.alpha = alpha

    def get_goal_hierarchy(self, int seq):
        assert seq < self.n_goals
        return [self.door_assignments[seq]]

    def get_complete_door_goal_hierarchy(self):
        cdef dict assgn
        return [dict(assgn) for assgn in self.door_assignments]

    def add_new_door_seq_context_assignment(self, int c, int seq, list hierarchy, GoalCluster goal_cluster):
        assert c % (self.n_sublvls+1) == 0
        assert seq <= self.n_sublvls
        cdef dict current_layer = hierarchy[0]
        cdef int k

        k = current_layer[c]
        # check if cluster "k" has already been assigned new cluster
        if k not in self.door_clusters[seq].keys():
            # if not, add an new mapping cluster
            assert goal_cluster is not None
            self.door_clusters[seq][k] = goal_cluster

        self.door_assignments[seq][c] = k
        self._door_log_prior[seq] = get_prior_log_probability(self.door_assignments[seq], self.alpha)
        
    def get_goal_probability(self, int c, int seq):
        assert seq >= 0 and seq <= self.n_sublvls
        cdef int k = self.door_assignments[seq][c]
        cdef GoalCluster cluster = self.door_clusters[seq][k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)
        cdef double [:] rew_func = cluster.goal_reward_probability

        cdef int g
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability

    def get_obs_goal_likelihood(self, int c, int goal, int r, int seq):
        assert seq >= 0 and seq <= self.n_sublvls
        cdef int k = self.door_assignments[seq][c]
        cdef GoalCluster cluster = self.door_clusters[seq][k]
        return cluster.get_observation_probability(goal, r)

    def find_goal_cluster(self, int c, int seq):
        assert seq >= 0 and seq <= self.n_sublvls
        cdef int k = self.door_assignments[seq][c]
        return self.door_clusters[seq][k]

    @property
    def log_prior(self):
        return np.sum(self._door_log_prior)
        
    def update(self, int c, int goal, int r, int seq):
        assert c % (self.n_sublvls+1) == 0
        assert seq >= 0 and seq <= self.n_sublvls
        cdef int k = self.door_assignments[seq][c]
        cdef GoalCluster cluster = self.door_clusters[seq][k]
        cluster.update(goal, r)

    def deep_copy(self, dict old_clusters, dict new_clusters):
        cdef DoorSeqCluster _cluster_copy = DoorSeqCluster(self.n_goals, self.alpha)
        
        cdef int ii, k, k2
        cdef dict old_dict
        cdef GoalCluster door_cluster, door_cluster_old
        _cluster_copy.door_assignments = [dict(old_dict) for old_dict in self.door_assignments]
        _cluster_copy._door_log_prior = np.array(self._door_log_prior)
        
        for ii in range(self.n_goals):
            for k, door_cluster in self.door_clusters[ii].iteritems():
                for k2, door_cluster_old in old_clusters.iteritems():
                    if door_cluster is door_cluster_old:
                        _cluster_copy.door_clusters[ii][k] = new_clusters[k2]
                        break

        return _cluster_copy
        
        
cdef class RoomCluster(object):

    cdef int n_goals, n_primitive_actions, n_abstract_actions
    cdef double gamma, iteration_criterion, inverse_temperature, alpha1
    cdef dict goal_assignments, goal_clusters, mapping_assignments, mapping_clusters
    cdef double _goal_log_prior, _mapping_log_prior

    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions, 
                 float inverse_temp, float gamma, float stop_criterion, float alpha1):

        self.n_goals = n_goals
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.alpha1 = alpha1
        
        self.mapping_assignments = dict()
        self.mapping_clusters = dict()
        self._mapping_log_prior = 0.
        
        self.goal_assignments = dict()
        self.goal_clusters = dict()
        self._goal_log_prior = 0.
        
    # goal update
    def update(self, int c, int goal, int r):
        cdef int k = self.goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]
        cluster.update(goal, r)
        
    def updating_mapping(self, int c, int a, int aa):
        cdef int k = self.mapping_assignments[c]
        cdef MappingCluster cluster = self.mapping_clusters[k]
        cluster.update(a, aa)
    
    def get_goal_probability(self, int c):
        cdef int k = self.goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)
        cdef double [:] rew_func = cluster.goal_reward_probability

        cdef int g
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability

    def get_mapping_probability(self, int c, int a, int aa):
        cdef MappingCluster cluster = self.mapping_clusters[self.mapping_assignments[c]]
        return cluster.mapping_mle[a, aa]
    
    def get_obs_goal_likelihood(self, int c, int goal, int r):
        cdef int k = self.goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]
        return cluster.get_observation_probability(goal, r)

    def get_obs_mapping_likelihood(self, int c, int a, int aa):
        cdef int k = self.mapping_assignments[c]
        cdef MappingCluster cluster = self.mapping_clusters[k]
        return cluster.get_likelihood(a, aa)

    def find_goal_cluster(self, int c):
        cdef int k = self.goal_assignments[c]
        return self.goal_clusters[k]

    def find_mapping_cluster(self, int c):
        cdef int k = self.mapping_assignments[c]
        return self.mapping_clusters[k]

    def get_goal_assignments(self):
        return [self.goal_assignments]
    
    def get_mapping_assignments(self):
        return [self.mapping_assignments]

    def add_new_goal_context_assignment(self, int c, list hierarchy, GoalCluster cluster):
        cdef dict upper_layer = hierarchy[0]
        cdef int k

        k = upper_layer[c]
        
        # check if cluster "k" has already been assigned new cluster
        if k not in self.goal_clusters.keys():
            # if not, add an new goal cluster
            assert cluster is not None
            self.goal_clusters[k] = cluster

        self.goal_assignments[c] = k  # note, there's no check built in here
        self._goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha1)
        
    def add_new_mapping_context_assignment(self, int c, list hierarchy, MappingCluster cluster):
        cdef dict layer = hierarchy[0]
        cdef int k = layer[c]

        # check if cluster "k" has already been assigned new cluster
        if k not in self.mapping_clusters.keys():
            # if not, add an new mapping cluster
            assert cluster is not None
            self.mapping_clusters[k] = cluster

        self.mapping_assignments[c] = k  # note, there's no check built in here
        self._mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha1)

    @property
    def goal_log_prior(self):
        return self._goal_log_prior

    @property
    def mapping_log_prior(self):
        return self._mapping_log_prior
    
    @property
    def log_prior(self):
        return self._goal_log_prior + self._mapping_log_prior
    
    def deep_copy(self, dict old_goal_clusters, dict new_goal_clusters, dict old_mapping_clusters, dict new_mapping_clusters):
        cdef int k, k2
        cdef GoalCluster goal_cluster, goal_cluster_old
        cdef MappingCluster mapping_cluster, mapping_cluster_old
        
        cdef RoomCluster _cluster_copy = RoomCluster(self.n_goals, self.n_primitive_actions, 
                                                self.n_abstract_actions, self.inverse_temperature, 
                                                self.gamma, self.iteration_criterion, self.alpha1)

        _cluster_copy._goal_log_prior = self._goal_log_prior 
        _cluster_copy._mapping_log_prior = self._mapping_log_prior

        _cluster_copy.goal_assignments = dict(self.goal_assignments)
        _cluster_copy.mapping_assignments = dict(self.mapping_assignments)
        
        for k, goal_cluster in self.goal_clusters.iteritems():
            for k2, goal_cluster_old in old_goal_clusters.iteritems():
                if goal_cluster is goal_cluster_old:
                    _cluster_copy.goal_clusters[k] = new_goal_clusters[k2]
                    break

        for k, mapping_cluster in self.mapping_clusters.iteritems():
            for k2, mapping_cluster_old in old_mapping_clusters.iteritems():
                if mapping_cluster is mapping_cluster_old:
                    _cluster_copy.mapping_clusters[k] = new_mapping_clusters[k2]
                    break

        return _cluster_copy
    
        
cdef class UpperRoomCluster(RoomCluster):
    cdef int n_sublvls
    cdef double alpha0
    cdef dict subroom_assignments, subroom_clusters, door_seq_assignments, door_seq_clusters
    cdef double _subroom_log_prior, _door_seq_log_prior
    cdef double [:] _sublvl_subroom_log_prior
    cdef list sublvl_subroom_assignments, sublvl_subroom_clusters

    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions, 
                 float inverse_temp, float gamma, float stop_criterion, float alpha0,
                 float alpha1):

        super(UpperRoomCluster, self).__init__(n_goals, n_primitive_actions, n_abstract_actions, 
                 inverse_temp, gamma, stop_criterion, alpha1)
        
        self.n_sublvls = n_goals-1
        self.alpha0 = alpha0
        
        self.subroom_assignments = dict()
        self.subroom_clusters = dict()
        self._subroom_log_prior = 0.

        self.door_seq_assignments = dict()
        self.door_seq_clusters = dict()
        self._door_seq_log_prior = 0.

        cdef int ii
        self.sublvl_subroom_assignments = [dict() for ii in range(self.n_sublvls)]
        self.sublvl_subroom_clusters = [dict() for ii in range(self.n_sublvls)]
        self._sublvl_subroom_log_prior = np.zeros(self.n_sublvls, dtype=DTYPE)
        

    def get_subroom_hierarchy(self, int c):
        cdef int sublvl = (c % (self.n_sublvls+1) ) - 1
        return [self.subroom_assignments, self.sublvl_subroom_assignments[sublvl]]
    
    def get_complete_subroom_hierarchy(self):
        cdef dict assgn
        return [self.subroom_assignments, [ dict(assgn) for assgn in self.sublvl_subroom_assignments]]
    
    def add_new_subroom_context_assignment(self, int c, list hierarchy, RoomCluster cluster):
        cdef dict layer = hierarchy[0]
        cdef int k
        cdef int sublvl = (c % (self.n_sublvls+1) ) - 1

        # check if new cluster draws from top layer of hierarchy
        if len(layer) > 0:
            k = layer[c]
        
            # check if cluster "k" has already been assigned new cluster
            if k not in self.subroom_clusters.keys():
                # if not, add an new subroom cluster
                assert cluster is not None
                self.subroom_clusters[k] = cluster

            self.subroom_assignments[c] = k  # note, there's no check built in here
            self._subroom_log_prior = get_prior_log_probability(self.subroom_assignments, self.alpha0)
            cluster = self.subroom_clusters[k]

        layer = hierarchy[1]
        k = layer[c]

        # check if cluster "k" has already been assigned new cluster
        if k not in self.sublvl_subroom_clusters[sublvl].keys():
            # if not, add an new subroom cluster
            assert cluster is not None
            self.sublvl_subroom_clusters[sublvl][k] = cluster

        self.sublvl_subroom_assignments[sublvl][c] = k  # note, there's no check built in here
        self._sublvl_subroom_log_prior[sublvl] = get_prior_log_probability(self.sublvl_subroom_assignments[sublvl], self.alpha1)

    def get_subroom_mapping_assignments(self, int c):
        cdef int sublvl = (c % (self.n_sublvls+1) ) - 1
        assert sublvl >= 0
        cdef int k = self.sublvl_subroom_assignments[sublvl][c]
        cdef RoomCluster cluster = self.sublvl_subroom_clusters[sublvl][k]

        return cluster.get_mapping_assignments()
    
    def get_complete_subroom_mapping_assignments(self):
        cdef RoomCluster cluster
        return [ [ cluster.get_mapping_assignments()[0] for cluster in sublvl_assgn.values() ] for sublvl_assgn in self.sublvl_subroom_clusters ]
    
    def add_new_subroom_mapping_context_assignment(self, int c, list hierarchy, MappingCluster new_cluster):
        cdef int sublvl = (c % (self.n_sublvls+1) ) - 1
        assert sublvl >= 0
        cdef int k = self.sublvl_subroom_assignments[sublvl][c]
        cdef RoomCluster cluster = self.sublvl_subroom_clusters[sublvl][k]

        return cluster.add_new_mapping_context_assignment(c, hierarchy, new_cluster)

    def get_door_seq_hierarchy(self):
        return [self.door_seq_assignments]

    def add_new_door_seq_context_assignment(self, int c, list hierarchy, DoorSeqCluster door_cluster):
        assert c % (self.n_sublvls+1) == 0
        cdef dict current_layer = hierarchy[0]
        cdef int k = current_layer[c]

        # check if cluster "k" has already been assigned new cluster
        if k not in self.door_seq_clusters.keys():
            # if not, add an new mapping cluster
            assert door_cluster is not None
            self.door_seq_clusters[k] = door_cluster

        self.door_seq_assignments[c] = k  # note, there's no check built in here
        self._door_seq_log_prior = get_prior_log_probability(self.door_seq_assignments, self.alpha1)

    def get_door_goal_hierarchy(self, int c, int seq):
        assert c % (self.n_sublvls+1) == 0
        assert seq <= self.n_sublvls
        cdef int k = self.door_seq_assignments[c]
        cdef DoorSeqCluster cluster = self.door_seq_clusters[k]
        
        return cluster.get_goal_hierarchy(seq)

    def get_complete_door_goal_hierarchy(self):
        cdef DoorSeqCluster cluster
        return [ cluster.get_complete_door_goal_hierarchy() for cluster in self.door_seq_clusters.values() ]
    
    def add_new_door_goal_context_assignment(self, int c, int seq, list hierarchy, GoalCluster goal_cluster):
        assert c % (self.n_sublvls+1) == 0
        assert seq <= self.n_sublvls
        cdef int k = self.door_seq_assignments[c]
        cdef DoorSeqCluster cluster = self.door_seq_clusters[k]
        cluster.add_new_door_seq_context_assignment(c, seq, hierarchy, goal_cluster)

    def get_subroom_goal_assignments(self, int c):
        cdef int sublvl = (c % (self.n_sublvls+1) ) - 1
        assert sublvl >= 0
        cdef int k = self.sublvl_subroom_assignments[sublvl][c]
        cdef RoomCluster cluster = self.sublvl_subroom_clusters[sublvl][k]

        return cluster.get_goal_assignments()

    def get_complete_subroom_goal_hierarchy(self):
        cdef RoomCluster cluster
        cdef dict sublvl_assgn
        return [ [ cluster.get_goal_assignments()[0] for cluster in sublvl_assgn.values() ] for sublvl_assgn in self.sublvl_subroom_clusters ]
        
    def add_new_subroom_goal_context_assignment(self, int c, list hierarchy, GoalCluster new_cluster):
        cdef int sublvl = (c % (self.n_sublvls+1) ) - 1
        assert sublvl >= 0
        cdef int k = self.sublvl_subroom_assignments[sublvl][c]
        cdef RoomCluster cluster = self.sublvl_subroom_clusters[sublvl][k]

        return cluster.add_new_goal_context_assignment(c, hierarchy, new_cluster)

    def update(self, int c, int goal, int r, int seq=-1):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        
        if sublvl < 0:
            assert seq >= 0 and seq <= self.n_sublvls
            k = self.door_seq_assignments[c]
            self.door_seq_clusters[k].update(c, goal, r, seq)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            self.sublvl_subroom_clusters[sublvl][k].update(c, goal, r)

    def updating_mapping(self, int c, int a, int aa):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        cdef RoomCluster cluster
        cdef int k

        if sublvl < 0:
            super(UpperRoomCluster, self).updating_mapping(c, a, aa)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            cluster = self.sublvl_subroom_clusters[sublvl][k]
            cluster.updating_mapping(c, a, aa)
            
    def get_goal_probability(self, int c, int seq=-1):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        
        if sublvl < 0:
            assert seq >= 0 and seq <= self.n_sublvls
            k = self.door_seq_assignments[c]
            return self.door_seq_clusters[k].get_goal_probability(c, seq)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            return self.sublvl_subroom_clusters[sublvl][k].get_goal_probability(c)

    def get_mapping_probability(self, int c, int a, int aa):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        cdef RoomCluster cluster
        cdef int k

        if sublvl < 0:
            return super(UpperRoomCluster, self).get_mapping_probability(c, a, aa)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            cluster = self.sublvl_subroom_clusters[sublvl][k]
            return cluster.get_mapping_probability(c, a, aa)

    def find_goal_cluster(self, int c, int seq):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        
        if sublvl < 0:
            assert seq >= 0 and seq <= self.n_sublvls
            k = self.door_seq_assignments[c]
            return self.door_seq_clusters[k].find_goal_cluster(c, seq)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            return self.sublvl_subroom_clusters[sublvl][k].find_goal_cluster(c)
        
    def find_mapping_cluster(self, int c):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        cdef RoomCluster cluster
        cdef int k

        if sublvl < 0:
            return super(UpperRoomCluster, self).find_mapping_cluster(c)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            cluster = self.sublvl_subroom_clusters[sublvl][k]
            return cluster.find_mapping_cluster(c)

    def get_obs_goal_likelihood(self, int c, int goal, int r, int seq=-1):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        
        if sublvl < 0:
            assert seq >= 0 and seq <= self.n_sublvls
            k = self.door_seq_assignments[c]
            return self.door_seq_clusters[k].get_obs_goal_likelihood(c, goal, r, seq)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            return self.sublvl_subroom_clusters[sublvl][k].get_obs_goal_likelihood(c, goal, r)

    def get_obs_mapping_likelihood(self, int c, int a, int aa):
        cdef int sublvl = ( c % (self.n_sublvls+1) ) - 1
        cdef RoomCluster cluster
        cdef int k

        if sublvl < 0:
            return super(UpperRoomCluster, self).get_obs_mapping_likelihood(c, a, aa)
        else:
            k = self.sublvl_subroom_assignments[sublvl][c]
            cluster = self.sublvl_subroom_clusters[sublvl][k]
            return cluster.get_obs_mapping_likelihood(c, a, aa)
            
    @property
    def log_prior(self):
        return self._door_seq_log_prior + self._mapping_log_prior + self._subroom_log_prior + np.sum(self._sublvl_subroom_log_prior)

    def deep_copy(self, dict old_goal_clusters, dict new_goal_clusters, dict old_mapping_clusters, dict new_mapping_clusters, 
                  dict old_subroom_clusters, dict new_subroom_clusters, dict old_door_seq_clusters, dict new_door_seq_clusters):

        cdef int ii, k, k2
        cdef dict old_dict
        cdef RoomCluster subroom_cluster, subroom_cluster_old
        cdef DoorSeqCluster door_seq_cluster, door_seq_cluster_old
        cdef GoalCluster goal_cluster, goal_cluster_old
        cdef MappingCluster mapping_cluster, mapping_cluster_old
        
        cdef UpperRoomCluster _cluster_copy = UpperRoomCluster(self.n_goals, self.n_primitive_actions, 
                                                self.n_abstract_actions, self.inverse_temperature, 
                                                self.gamma, self.iteration_criterion, self.alpha0, self.alpha1)

        _cluster_copy._goal_log_prior = self._goal_log_prior 
        _cluster_copy._mapping_log_prior = self._mapping_log_prior

        _cluster_copy.goal_assignments = dict(self.goal_assignments)
        _cluster_copy.mapping_assignments = dict(self.mapping_assignments)
        
        _cluster_copy.subroom_assignments = dict(self.subroom_assignments)
        _cluster_copy._subroom_log_prior = self._subroom_log_prior

        _cluster_copy.door_seq_assignments = dict(self.door_seq_assignments)
        _cluster_copy._door_seq_log_prior = self._door_seq_log_prior
        
        _cluster_copy.sublvl_subroom_assignments = [dict(old_dict) for old_dict in self.sublvl_subroom_assignments]
        _cluster_copy._sublvl_subroom_log_prior = np.array(self._sublvl_subroom_log_prior)

        for k, goal_cluster in self.goal_clusters.iteritems():
            for k2, goal_cluster_old in old_goal_clusters.iteritems():
                if goal_cluster is goal_cluster_old:
                    _cluster_copy.goal_clusters[k] = new_goal_clusters[k2]
                    break

        for k, mapping_cluster in self.mapping_clusters.iteritems():
            for k2, mapping_cluster_old in old_mapping_clusters.iteritems():
                if mapping_cluster is mapping_cluster_old:
                    _cluster_copy.mapping_clusters[k] = new_mapping_clusters[k2]
                    break

        for k, subroom_cluster in self.subroom_clusters.iteritems():
            for k2, subroom_cluster_old in old_subroom_clusters.iteritems():
                if subroom_cluster is subroom_cluster_old:
                    _cluster_copy.subroom_clusters[k] = new_subroom_clusters[k2]
                    break

        for k, door_seq_cluster in self.door_seq_clusters.iteritems():
            for k2, door_seq_cluster_old in old_door_seq_clusters.iteritems():
                if door_seq_cluster is door_seq_cluster_old:
                    _cluster_copy.door_seq_clusters[k] = new_door_seq_clusters[k2]
                    break

        for ii in range(self.n_sublvls):
            for k, subroom_cluster in self.sublvl_subroom_clusters[ii].iteritems():
                for k2, subroom_cluster_old in old_subroom_clusters.iteritems():
                    if subroom_cluster is subroom_cluster_old:
                        _cluster_copy.sublvl_subroom_clusters[ii][k] = new_subroom_clusters[k2]
                        break

        return _cluster_copy

    
cdef class HierarchicalHypothesis(object):
    cdef int n_goals, n_sublvls, n_primitive_actions, n_abstract_actions
    cdef double gamma, iteration_criterion, inverse_temperature, alpha0, alpha1
    cdef dict upper_room_clusters, subroom_clusters, door_seq_clusters, goal_clusters, mapping_clusters
    cdef dict upper_room_assignments, subroom_assignments, door_seq_assignments, goal_assignments, mapping_assignments
    cdef double _upper_room_log_prior, _subroom_log_prior, _door_seq_log_prior, _goal_log_prior, _mapping_log_prior
    cdef double _total_log_prior
    cdef double [:] _goal_log_likelihoods, _mapping_log_likelihoods
    cdef double goal_prior, mapping_prior

    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions, 
                 float inverse_temp, float gamma, float stop_criterion, float alpha0, 
                 float alpha1, float goal_prior, float mapping_prior):

        self.n_goals = n_goals
        self.n_sublvls = n_goals-1
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
        self.upper_room_assignments = dict()
        self.upper_room_clusters = dict()
        self._upper_room_log_prior = 0.

        self.subroom_assignments = dict()
        self.subroom_clusters = dict()
        self._subroom_log_prior = 0.

        self.door_seq_assignments = dict()
        self.door_seq_clusters = dict()
        self._door_seq_log_prior = 0.

        self.goal_assignments = dict()
        self.goal_clusters = dict()
        self._goal_log_prior = 0.

        self.mapping_clusters = dict()
        self.mapping_assignments = dict()
        self._mapping_log_prior = 0.
        
        self.goal_prior = goal_prior
        self.mapping_prior = mapping_prior

        # note: cluster keys must be in consecutive order starting from 0
        # Aame keys will be used to index log likelihood arrays to retrieve corresponding log likelihood.
        self._goal_log_likelihoods = np.zeros(0, dtype=DTYPE)
        self._mapping_log_likelihoods = np.zeros(0, dtype=DTYPE)
        self._total_log_prior = 1.
        
    def get_upper_room_assignments(self):
        return self.upper_room_assignments

    def add_new_upper_room_context_assignment(self, int c, int k):
        assert c % (self.n_sublvls+1) == 0
        
        # check if cluster "k" is already been assigned new cluster
        if k not in self.upper_room_clusters.keys():
            # if not, add an new upper_room cluster
            self.upper_room_clusters[k] = UpperRoomCluster(self.n_goals, self.n_primitive_actions, 
                self.n_abstract_actions, self.inverse_temperature, self.gamma, self.iteration_criterion, 
                self.alpha0, self.alpha1)
            
        self.upper_room_assignments[c] = k  # note, there's no check built in here
        self._upper_room_log_prior = get_prior_log_probability(self.upper_room_assignments, self.alpha0)

    def get_subroom_hierarchy(self, int c):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        
        return [self.subroom_assignments] + cluster.get_subroom_hierarchy(c)

    def get_complete_subroom_hierarchy(self):
        cdef UpperRoomCluster cluster        
        return [self.subroom_assignments, [cluster.get_complete_subroom_hierarchy() for cluster in self.upper_room_clusters.values()]]

    def add_new_subroom_context_assignment(self, int c, list hierarchy):
        cdef dict upper_layer = hierarchy[0]
        cdef int k
        cdef RoomCluster cluster = None

        # check if new cluster draws from top layer of hierarchy
        if len(upper_layer) > 0:
            k = upper_layer[c]
        
            # check if cluster "k" has already been assigned new cluster
            if k not in self.subroom_clusters.keys():
                # if not, add new subroom cluster
                cluster = RoomCluster(self.n_goals, self.n_primitive_actions, self.n_abstract_actions, 
                                      self.inverse_temperature, self.gamma, self.iteration_criterion, 
                                      self.alpha1)
                self.subroom_clusters[k] = cluster

            self.subroom_assignments[c] = k  # note, there's no check built in here
            self._subroom_log_prior = get_prior_log_probability(self.subroom_assignments, self.alpha0)
            cluster = self.subroom_clusters[k]

        hierarchy = hierarchy[1:]
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        k = self.upper_room_assignments[upper_room_c]

        self.upper_room_clusters[k].add_new_subroom_context_assignment(c, hierarchy, cluster)
    
    def get_mapping_hierarchy(self, int c):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int sublvl = c - upper_room_c - 1
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]

        if sublvl < 0:
            return [self.mapping_assignments] + cluster.get_mapping_assignments()
        else:
            return [self.mapping_assignments] + cluster.get_subroom_mapping_assignments(c)

    def get_complete_mapping_hierarchy(self):
        cdef UpperRoomCluster cluster
        return [self.mapping_assignments, [ [cluster.get_mapping_assignments(), 
                                             cluster.get_complete_subroom_mapping_assignments()] for cluster in self.upper_room_clusters.values()] ]

    def add_new_mapping_context_assignment(self, int c, list hierarchy):
        cdef dict upper_layer = hierarchy[0]
        cdef MappingCluster cluster = None
        cdef int k

        if len(upper_layer) > 0:
            k = upper_layer[c]
        
            # check if cluster "k" has already been assigned new cluster
            if k not in self.mapping_clusters.keys():
                # if not, add an new mapping cluster
                cluster = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                         self.mapping_prior)
                self.mapping_clusters[k] = cluster
                self._mapping_log_likelihoods = np.concatenate((self._mapping_log_likelihoods,[0.]))

            self.mapping_assignments[c] = k  # note, there's no check built in here
            self._mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha0)
            cluster = self.mapping_clusters[k]
            
        hierarchy = hierarchy[1:]
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int sublvl = c - upper_room_c - 1
        k = self.upper_room_assignments[upper_room_c]

        if sublvl < 0:
            self.upper_room_clusters[k].add_new_mapping_context_assignment(c, hierarchy, cluster)
        else:
            self.upper_room_clusters[k].add_new_subroom_mapping_context_assignment(c, hierarchy, cluster)

    def get_door_seq_hierarchy(self, int c):
        assert c % (self.n_sublvls+1) == 0
        cdef int k = self.upper_room_assignments[c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        
        return [self.door_seq_assignments] + cluster.get_goal_assignments()

    def get_complete_door_seq_hierarchy(self):
        cdef UpperRoomCluster cluster
        return [self.door_seq_assignments, [ cluster.get_goal_assignments() for cluster in self.upper_room_clusters.values() ] ]

    def add_new_door_seq_context_assignment(self, int c, list hierarchy):
        assert c % (self.n_sublvls+1) == 0
        cdef dict current_layer = hierarchy[0]
        cdef DoorSeqCluster door_cluster = None
        cdef int k

        if len(current_layer) > 0:
            k = current_layer[c]
        
            # check if cluster "k" has already been assigned new cluster
            if k not in self.door_seq_clusters.keys():
                # if not, add an new mapping cluster
                door_cluster = DoorSeqCluster(self.n_goals, self.alpha1)
                self.door_seq_clusters[k] = door_cluster

            self.door_seq_assignments[c] = k  # note, there's no check built in here
            self._door_seq_log_prior = get_prior_log_probability(self.door_seq_assignments, self.alpha0)
            door_cluster = self.door_seq_clusters[k]

        hierarchy = hierarchy[1:]
        k = self.upper_room_assignments[c]
        self.upper_room_clusters[k].add_new_door_seq_context_assignment(c, hierarchy, door_cluster)

    def get_door_goal_hierarchy(self, int c, int seq):
        assert c % (self.n_sublvls+1) == 0
        assert seq <= self.n_sublvls
        cdef int k = self.upper_room_assignments[c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        
        return [self.goal_assignments] + cluster.get_door_goal_hierarchy(c, seq)

    def add_new_door_goal_context_assignment(self, int c, int seq, list hierarchy):
        assert c % (self.n_sublvls+1) == 0
        assert seq <= self.n_sublvls
        cdef dict current_layer = hierarchy[0]
        cdef GoalCluster goal_cluster = None
        cdef int k

        if len(current_layer) > 0:
            k = current_layer[(c,seq)]
        
            # check if cluster "k" has already been assigned new cluster
            if k not in self.goal_clusters.keys():
                # if not, add an new mapping cluster
                goal_cluster = GoalCluster(self.n_goals, self.goal_prior)
                self.goal_clusters[k] = goal_cluster
                self._goal_log_likelihoods = np.concatenate((self._goal_log_likelihoods,[0.]))

            self.goal_assignments[(c,seq)] = k  # note, there's no check built in here
            self._goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha0)
            goal_cluster = self.goal_clusters[k]

        hierarchy = hierarchy[1:]
        k = self.upper_room_assignments[c]
        self.upper_room_clusters[k].add_new_door_goal_context_assignment(c, seq, hierarchy, goal_cluster)

    def get_subroom_goal_hierarchy(self, int c):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int sublvl = c - upper_room_c - 1
        assert sublvl >= 0

        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]

        return [self.goal_assignments] + cluster.get_subroom_goal_assignments(c)

    def get_complete_goal_hierarchy(self):
        cdef UpperRoomCluster cluster
        return [self.goal_assignments, [ [ cluster.get_complete_door_goal_hierarchy(), cluster.get_complete_subroom_goal_hierarchy() ] for cluster in self.upper_room_clusters.values() ] ]

    def add_new_subroom_goal_context_assignment(self, int c, list hierarchy):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int sublvl = c - upper_room_c - 1
        assert sublvl >= 0

        cdef int k
        cdef dict upper_layer = hierarchy[0]
        cdef GoalCluster cluster = None

        if len(upper_layer) > 0:
            k = upper_layer[c]
        
            # check if cluster "k" has already been assigned new cluster
            if k not in self.goal_clusters.keys():
                # if not, add an new mapping cluster
                cluster = GoalCluster(self.n_goals, self.goal_prior)
                self.goal_clusters[k] = cluster
                self._goal_log_likelihoods = np.concatenate((self._goal_log_likelihoods,[0.]))

            self.goal_assignments[c] = k  # note, there's no check built in here
            self._goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha0)
            cluster = self.goal_clusters[k]
            
        hierarchy = hierarchy[1:]
        k = self.upper_room_assignments[upper_room_c]
        self.upper_room_clusters[k].add_new_subroom_goal_context_assignment(c, hierarchy, cluster)

    # goal update
    def update(self, int c, int goal, int r, int seq=-1):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        cluster.update(c, goal, r, seq)
        self.update_goal_log_likelihood(c, seq)

    def updating_mapping(self, int c, int a, int aa):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        cluster.updating_mapping(c, a, aa)
        self.update_mapping_log_likelihood(c)

    def get_goal_probability(self, int c, int seq=-1):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        return cluster.get_goal_probability(c, seq)

    def get_mapping_probability(self, int c, int a, int aa):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        return cluster.get_mapping_probability(c, a, aa)

    def update_goal_log_likelihood(self, int c, int seq=-1):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster upper_room_cluster = self.upper_room_clusters[k]
        cdef GoalCluster goal_cluster = upper_room_cluster.find_goal_cluster(c, seq)
        
        k = 0
        while goal_cluster is not self.goal_clusters[k]:
            k += 1
        self._goal_log_likelihoods[k] = goal_cluster.get_log_likelihood()

    def update_mapping_log_likelihood(self, int c):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster upper_room_cluster = self.upper_room_clusters[k]
        cdef MappingCluster map_cluster = upper_room_cluster.find_mapping_cluster(c)
        
        k = 0
        while map_cluster is not self.mapping_clusters[k]:
            k += 1
        self._mapping_log_likelihoods[k] = map_cluster.log_likelihood()

    def get_obs_goal_likelihood(self, int c, int goal, int r, int seq=-1):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        return cluster.get_obs_goal_likelihood(c, goal, r, seq)

    def get_obs_mapping_likelihood(self, int c, int a, int aa):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int k = self.upper_room_assignments[upper_room_c]
        cdef UpperRoomCluster cluster = self.upper_room_clusters[k]
        return cluster.get_obs_mapping_likelihood(c, a, aa)

    def get_log_posterior(self):
        # To calculate log posterior, first consider the generative process for hypothesis:
        # 1) Upper Room CRP is first generated
        # 2) Hierarchical subroom CRP is generated next.
        # 3) Hierarchcal mapping CRP generated next.
        # 4) Hierarchical goal CRP (including in DoorSeq clusters) generated.
        # 5) Goal and mapping observations generated from given latent hierarchy.
        # Posterior probability must be given by probability of generating this hypothesis.
        # Can regard CRP at each layer of hierarchy as being generated independently of all other CRPs.
        # Therefore each CRP contributes to posterior independently of other CRPs or hierarchies.
        # Example: add ctx 1 to subroom mapping CRP via upper room A and add ctx 2 to same CRP via upper room B.
        # Prob of the mapping CRP will not depend on through which upper rooms the ctxs entered mapping CRP.
        # That mapping CRP will expand as a CRP that is independent of other hierarchies.
        
        return self._total_log_prior + np.sum(self._goal_log_likelihoods) + np.sum(self._mapping_log_likelihoods)
    
    def update_log_prior(self):
        cdef DoorSeqCluster door_cluster
        cdef UpperRoomCluster upper_room_cluster
        cdef RoomCluster subroom_cluster

        self._total_log_prior = self._upper_room_log_prior + self._subroom_log_prior + self._door_seq_log_prior + self._goal_log_prior + self._mapping_log_prior
        self._total_log_prior += sum([upper_room_cluster.log_prior for upper_room_cluster in self.upper_room_clusters.values()])
        self._total_log_prior += sum([subroom_cluster.log_prior for subroom_cluster in self.subroom_clusters.values()])
        self._total_log_prior += sum([door_cluster.log_prior for door_cluster in self.door_seq_clusters.values()])

    def deep_copy(self):
        cdef HierarchicalHypothesis _h_copy = HierarchicalHypothesis(self.n_goals, 
                self.n_primitive_actions, self.n_abstract_actions, self.inverse_temperature, 
                self.gamma, self.iteration_criterion, self.alpha0, self.alpha1, self.goal_prior, self.mapping_prior)

        cdef GoalCluster goal_cluster
        cdef MappingCluster map_cluster
        cdef DoorSeqCluster door_cluster
        cdef UpperRoomCluster upper_room_cluster
        cdef RoomCluster subroom_cluster
        
        assert len(self._goal_log_likelihoods) == len(self.goal_clusters)
        assert len(self._mapping_log_likelihoods) == len(self.mapping_clusters)

        _h_copy._upper_room_log_prior = self._upper_room_log_prior
        _h_copy._subroom_log_prior = self._subroom_log_prior
        _h_copy._door_seq_log_prior = self._door_seq_log_prior
        _h_copy._goal_log_prior = self._goal_log_prior
        _h_copy._mapping_log_prior = self._mapping_log_prior

        _h_copy._total_log_prior = self._total_log_prior
        _h_copy._goal_log_likelihoods = np.array(self._goal_log_likelihoods)
        _h_copy._mapping_log_likelihoods = np.array(self._mapping_log_likelihoods)
        
        _h_copy.upper_room_assignments = dict(self.upper_room_assignments)
        _h_copy.subroom_assignments = dict(self.subroom_assignments)
        _h_copy.door_seq_assignments = dict(self.door_seq_assignments)
        _h_copy.goal_assignments = dict(self.goal_assignments)
        _h_copy.mapping_assignments = dict(self.mapping_assignments)
        
        _h_copy.goal_clusters = {k:goal_cluster.deep_copy() for k,goal_cluster in self.goal_clusters.iteritems()}
        _h_copy.mapping_clusters = {k:map_cluster.deep_copy() for k,map_cluster in self.mapping_clusters.iteritems()}
        _h_copy.door_seq_clusters = {k:door_cluster.deep_copy(self.goal_clusters, _h_copy.goal_clusters) for k,door_cluster in self.door_seq_clusters.iteritems()}
        _h_copy.subroom_clusters = {k:cluster.deep_copy(self.goal_clusters, _h_copy.goal_clusters,
                                                        self.mapping_clusters, _h_copy.mapping_clusters) for k,cluster in self.subroom_clusters.iteritems()}
        _h_copy.upper_room_clusters = {k:upper_room_cluster.deep_copy(self.goal_clusters, _h_copy.goal_clusters, 
                                                                      self.mapping_clusters, _h_copy.mapping_clusters, 
                                                                      self.subroom_clusters, _h_copy.subroom_clusters,
                                                                      self.door_seq_clusters, _h_copy.door_seq_clusters) for k,upper_room_cluster in self.upper_room_clusters.iteritems()}

        return _h_copy

    def spawn_new_hypotheses(self, int c, int seq=-1):
        cdef int upper_room_c = c - ( c % (self.n_sublvls+1) )
        cdef int sublvl = c - upper_room_c - 1

        cdef list new_hypotheses = list()
        cdef list tmp_hypotheses1 = list()
        cdef list tmp_hypotheses2 = list()
        cdef list hierarchy
        cdef HierarchicalHypothesis h_new, h_tmp

        cdef int n_upper_room_clusters
        cdef list old_hierarchies = list()
        cdef list new_hierarchies = list()
        
        if len(self.upper_room_assignments) > 0:
            n_upper_room_clusters = max(self.upper_room_assignments.values()) + 2
        else:
            n_upper_room_clusters = 1
        
        cdef int k
        
        if sublvl < 0:
            assert seq >= 0 and seq <= self.n_sublvls

            # new context in upper room
            if seq == 0:
                # expand upper_room CRP
                for k in range(0, n_upper_room_clusters):
                    # for each possible context assignment to room clusters,
                    # create copy of current hypothesis and add room cluster to it
                    h_new = self.deep_copy()
                    h_new.add_new_upper_room_context_assignment(c, k)
                    tmp_hypotheses2.append(h_new)
                
                # add goals
                for h_tmp in tmp_hypotheses2:
                    old_hierarchy = h_tmp.get_door_seq_hierarchy(c)
                    new_hierarchies = augment_hierarchical_CRP(old_hierarchy, c)
                
                    for hierarchy in new_hierarchies:
                        h_new = h_tmp.deep_copy()
                        h_new.add_new_door_seq_context_assignment(c, hierarchy)
                        tmp_hypotheses1.append(h_new)

                tmp_hypotheses2 = list()
                for h_tmp in tmp_hypotheses1:
                    old_hierarchy = h_tmp.get_door_goal_hierarchy(c, seq)
                    new_hierarchies = augment_hierarchical_door_CRP(old_hierarchy, c, seq)
                
                    for hierarchy in new_hierarchies:
                        h_new = h_tmp.deep_copy()
                        h_new.add_new_door_goal_context_assignment(c, seq, hierarchy)
                        tmp_hypotheses2.append(h_new)

            else: # old upper room context, but new sequence
                
                # only need to add goals (done here) and mappings (done below)
                old_hierarchy = self.get_door_goal_hierarchy(c, seq)
                new_hierarchies = augment_hierarchical_door_CRP(old_hierarchy, c, seq)
                
                for hierarchy in new_hierarchies:
                    h_new = self.deep_copy()
                    h_new.add_new_door_goal_context_assignment(c, seq, hierarchy)
                    tmp_hypotheses2.append(h_new)
        else:
            old_hierarchy = self.get_subroom_hierarchy(c)
            new_hierarchies = augment_hierarchical_CRP(old_hierarchy, c)
            
            for hierarchy in new_hierarchies:
                h_new = self.deep_copy()
                h_new.add_new_subroom_context_assignment(c, hierarchy)
                tmp_hypotheses1.append(h_new)
                
            # add goals
            for h_tmp in tmp_hypotheses1:
                old_hierarchy = h_tmp.get_subroom_goal_hierarchy(c)
                new_hierarchies = augment_hierarchical_CRP(old_hierarchy, c)
                
                for hierarchy in new_hierarchies:
                    h_new = h_tmp.deep_copy()
                    h_new.add_new_subroom_goal_context_assignment(c, hierarchy)
                    tmp_hypotheses2.append(h_new)
                
        tmp_hypotheses1 = list()  # flush memory

        # add mappings if in new sublvl or first time in upper room
        if sublvl > -1 or seq == 0:
            for h_tmp in tmp_hypotheses2:
                old_hierarchy = h_tmp.get_mapping_hierarchy(c)
                new_hierarchies = augment_hierarchical_CRP(old_hierarchy, c)
                
                for hierarchy in new_hierarchies:
                    h_new = h_tmp.deep_copy()
                    h_new.add_new_mapping_context_assignment(c, hierarchy)
                    new_hypotheses.append(h_new)
        else:
            new_hypotheses = tmp_hypotheses2
                
        for h_tmp in new_hypotheses:
            h_tmp.update_log_prior()

        return new_hypotheses


cdef list augment_hierarchical_CRP(list old_hierarchy, int new_context):
    # output list of hierarchies, each layer of hierarchy contains single element
    # dict indicating which cluster context should be assigned to
    
    cdef int n_layers = len(old_hierarchy)
    cdef int ii
    
    cdef list new_hierarchies = list()
    cdef list new_layers = list()
    
    cdef list hierarchy, tmp_hierarchies
    cdef dict old_layer, assignment
    
    # generate different possible context assignments for each layer
    for ii, old_layer in enumerate(old_hierarchy):
        new_layers.append( augment_CRP(old_layer, new_context) )
        
    # "stitch" the context assignments for the different layers together into single consistent hierarchies
    tmp_hierarchies = new_layers[-1]
    for ii in range(n_layers-2,-1,-1):
        tmp_hierarchies = combine_hierarchy_layers(new_layers[ii], tmp_hierarchies)
        
    new_hierarchies += tmp_hierarchies
        
    return new_hierarchies
            
            
cdef list augment_CRP(dict old_assignments, int new_context):
    cdef int n_clusters
    
    if len(old_assignments) > 0:
        n_clusters = max(old_assignments.values())+2
    else:
        n_clusters = 1
    
    cdef list new_assignments = [[{new_context: k}] for k in range(n_clusters)]
    return new_assignments


cdef list combine_hierarchy_layers(list upper_layer, list lower_layer):
    cdef list new_hierarchies = list()
    cdef list hierarchy
    cdef list upper_assignment
    
    for hierarchy in lower_layer[:-1]:
        new_hierarchies.append( [{}] + hierarchy )
        # note: shallow copy of hierarchy to save mem
    
    for upper_assignment in upper_layer:
        new_hierarchies.append( upper_assignment + lower_layer[-1] )
        
    return new_hierarchies


cdef list augment_hierarchical_door_CRP(list old_hierarchy, int new_context, int seq):
    cdef int n_layers = len(old_hierarchy)
    cdef int ii
    
    cdef list new_hierarchies = list()
    cdef list new_layers = list()
    
    cdef list hierarchy, tmp_hierarchies
    cdef dict old_layer, assignment
    
    # generate different possible context assignments for each layer
    # Top most layer is always the env-wide goal CRP.
    # For upper room contexts, (context, seq) get assigned to the clusters of this CRP.
    new_layers.append( augment_CRP_with_tuple(old_hierarchy[0], (new_context, seq) ) )
    
    for ii, old_layer in enumerate(old_hierarchy[1:]):
        new_layers.append( augment_CRP(old_layer, new_context) )
        
    # "stitch" the context assignments for the different layers together into single consistent hierarchies
    tmp_hierarchies = new_layers[-1]
    for ii in range(n_layers-2,-1,-1):
        tmp_hierarchies = combine_hierarchy_layers(new_layers[ii], tmp_hierarchies)
        
    new_hierarchies += tmp_hierarchies
        
    return new_hierarchies


cdef list augment_CRP_with_tuple(dict old_assignments, tuple new_context):
    cdef int n_clusters
    
    if len(old_assignments) > 0:
        n_clusters = max(old_assignments.values())+2
    else:
        n_clusters = 1
    
    cdef list new_assignments = [[{new_context: k}] for k in range(n_clusters)]
    return new_assignments
    







cdef class UpperDoorHypothesis(object):
    cdef int n_goals, n_sublvls
    cdef double alpha, goal_prior
    cdef list cluster_assignments, clusters, log_likelihood, log_prior

    def __init__(self, int n_goals, float alpha, float goal_prior):
        cdef int ii
        
        self.n_goals = n_goals
        self.n_sublvls = n_goals-1

        self.cluster_assignments = [dict() for ii in range(self.n_sublvls+1)]
        self.clusters = [dict() for ii in range(self.n_sublvls+1)]
        self.log_likelihood = [dict() for ii in range(self.n_sublvls+1)]

        self.alpha = alpha
        self.goal_prior = goal_prior
        self.log_prior = [0. for ii in range(self.n_sublvls+1)]

    def update(self, int c, int seq, int goal, int r):
        assert c % (self.n_sublvls+1) == 0
        assert seq >= 0 and seq <= self.n_sublvls
        cdef int k = self.cluster_assignments[seq][c]
        cdef GoalCluster cluster = self.clusters[seq][k]
        cluster.update(goal, r)
        self.log_likelihood[seq][k] = cluster.get_log_likelihood()

    def get_log_likelihood(self):
        cdef double log_likelihood = 0, tmp
        cdef int ii
        
        for ii in range(len(self.log_likelihood)):
            log_likelihood += sum(self.log_likelihood[ii].values())
        
        return log_likelihood
    
    def get_obs_likelihood(self, int c, int seq, int goal, int r):
        assert seq >= 0 and seq <= self.n_sublvls
        cdef int k = self.cluster_assignments[seq][c]
        cdef GoalCluster cluster = self.clusters[seq][k]
        return cluster.get_observation_probability(goal, r)

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.get_log_prior()

    def get_log_prior(self):
        return sum(self.log_prior)

    def get_goal_probability(self, int c, int seq):
        assert seq >= 0 and seq <= self.n_sublvls
        cdef int k = self.cluster_assignments[seq][c]
        cdef GoalCluster cluster = self.clusters[seq][k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)
        cdef double [:] rew_func = cluster.goal_reward_probability

        cdef int g
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability

    def deep_copy(self):
        cdef UpperDoorHypothesis _h_copy = UpperDoorHypothesis(self.n_goals, self.alpha, self.goal_prior)

        _h_copy.log_prior = list(self.log_prior)

        cdef dict assignments, seq_clusters, log_likelihoods
        cdef GoalCluster cluster
        _h_copy.cluster_assignments = [dict(assignments) for assignments in self.cluster_assignments]
        _h_copy.log_likelihood = [dict(log_likelihoods) for log_likelihoods in self.log_likelihood]
        _h_copy.clusters = [{k : cluster.deep_copy() for k,cluster in seq_clusters.iteritems()} for seq_clusters in self.clusters]

        return _h_copy

    def get_assignments(self, int seq):
        return self.cluster_assignments[seq]

    def add_new_context_assignment(self, int c, int k, int seq):
        assert seq >= 0 and seq <= self.n_sublvls
        cdef dict assignments = self.cluster_assignments[seq]
        
        # check if cluster "k" is already been assigned new cluster
        if k not in assignments.values():
            # if not, add an new reward cluster
            self.clusters[seq][k] = GoalCluster(self.n_goals, self.goal_prior)
            self.log_likelihood[seq][k] = 0.

        assignments[c] = k  # note, there's no check built in here
        self.log_prior[seq] = get_prior_log_probability(assignments, self.alpha)
        