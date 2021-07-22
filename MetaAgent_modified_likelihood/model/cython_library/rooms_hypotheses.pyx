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

        for a in range(self.n_primitive_actions):
            _cluster_copy.primitive_action_counts[a] = self.primitive_action_counts[a]

            for aa in range(self.n_abstract_actions + 1): # include the possibility of the "wait" action
                _cluster_copy.mapping_history[a, aa] = self.mapping_history[a, aa]
                _cluster_copy.mapping_mle[a, aa] = self.mapping_mle[a, aa]
                _cluster_copy.pr_aa_given_a[a, aa] = self.pr_aa_given_a[a, aa]

        for aa in range(self.n_abstract_actions + 1): # include the possibility of the "wait" action
            _cluster_copy.abstract_action_counts[aa] = self.abstract_action_counts[aa]

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
            # if not, add an new reward cluster
            self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)

        self.cluster_assignments[c] = k
        self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)


cdef class GoalCluster(object):
    cdef int n_goals
    cdef double set_visits, goal_prior
    cdef double [:]  goal_rewards_received, goal_reward_probability

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

        for g in range(self.n_goals):
            _cluster_copy.set_visits = self.set_visits
            _cluster_copy.goal_rewards_received[g] = self.goal_rewards_received[g]
            _cluster_copy.goal_reward_probability[g] = self.goal_reward_probability[g]

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


cdef class RoomsHypothesis(object):
    cdef int n_goals, n_primitive_actions, n_abstract_actions
    cdef double room_log_prior, goal_log_prior, mapping_log_prior, total_log_prior, room_prior, goal_prior, mapping_prior, total_room_goal_log_prior, total_room_mapping_log_prior, alpha
    cdef dict room_assignments, room_clusters, goal_assignments, goal_clusters, mapping_assignments, mapping_clusters
    cdef list room_experience
    
    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions,
                 double alpha, double room_prior, double goal_prior, double mapping_prior):

        self.n_goals = n_goals
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.room_clusters = dict()
        self.goal_clusters = dict()
        self.mapping_clusters = dict()

        self.room_assignments = dict()
        self.goal_assignments = dict()
        self.mapping_assignments = dict()

        self.alpha = alpha
        self.room_prior = room_prior
        self.goal_prior = goal_prior
        self.mapping_prior = mapping_prior

        # initialize posterior
        self.room_log_prior = 1.0/4
        self.goal_log_prior = 1.0/4
        self.mapping_log_prior = 1.0/4
        self.room_goal_mapping_log_prior = 1.0/4
        self.total_log_prior = 1.0
        
    def add_new_context_assignment(self, int c, int k, list hierarchical_goal_assignment, list hierarchical_mapping_assignment):

        # check if cluster "k" has already been assigned new cluster
        if k not in self.room_assignments.values():
            # if not, add an new reward cluster
            self.room_clusters[k] = RoomCluster(self.n_goals, self.n_primitive_actions, self.n_abstract_actions, 
                              self.alpha, self.goal_prior, self.mapping_prior)

        self.room_assignments[c] = k  # note, there's no check built in here
        self.room_log_prior = get_prior_log_probability(self.room_assignments, self.alpha)
        
        cdef RoomCluster Room = self.room_clusters[k]
        
        # need to get log prior for each room cluster's goal and mapping crp's
        

        self.add_new_goal_assignment(c, hierarchical_goal_assignment, Room)
        self.add_new_mapping_assignment(c, hierarchical_mapping_assignment, Room)

        self.update_total_room_goal_log_prior()
        self.total_log_prior = self.room_log_prior + self.goal_log_prior + self.mapping_log_prior + self.room_goal_mapping_log_prior
        
    def add_new_goal_assignment(self, int c, list hierarchical_goal_assignment, RoomCluster Room):
        
        cdef int k, lower_k
        cdef GoalCluster cluster = None
        
        assert len(hierarchical_goal_assignment) == 2
        
        # check if new context has assignment in environment-wide CRP and expand CRP if needed
        cdef dict assignments = hierarchical_goal_assignment[0]
        if c in assignments.keys():
            k = assignments[c]
            
            # check if cluster "k" is already been assigned new cluster
            if k not in self.goal_clusters.keys():
                # if not, add an new reward cluster
                self.goal_clusters[k] = GoalCluster(self.n_goals, self.goal_prior)
            
            self.goal_assignments[c] = k
        
            self.goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha)
            cluster = self.goal_clusters[k]
        
        lower_k = hierarchical_goal_assignment[1][c]
        Room.add_new_goal_assignment(c, lower_k, cluster)
        
    def add_new_mapping_assignment(self, int c, list hierarchical_mapping_assignment, RoomCluster Room):
        
        cdef int k, lower_k
        cdef MappingCluster cluster = None
        
        assert len(hierarchical_mapping_assignment) == 2
        
        # check if new context has assignment in environment-wide CRP and expand CRP if needed
        cdef dict assignments = hierarchical_mapping_assignment[0]
        if c in assignments.keys():
            k = assignments[c]
            
            # check if cluster "k" is already been assigned new cluster
            if k not in self.mapping_clusters.keys():
                # if not, add an new reward cluster
                self.mapping_clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)
            
            self.mapping_assignments[c] = k
        
            self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha)
            cluster = self.mapping_clusters[k]
        
        lower_k = hierarchical_mapping_assignment[1][c]
        Room.add_new_mapping_assignment(c, lower_k, cluster)
        
    def update_total_room_goal_log_prior(self):
        self.room_goal_mapping_log_prior = sum([ Room.get_room_goal_mapping_log_prior() for Room in self.room_clusters.values() ])

    def update_mapping(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef RoomCluster cluster = self.room_clusters[k]

        cluster.update_mapping(c, a, aa)

    def update(self, int c, int goal, int r):
        cdef int k = self.room_assignments[c]
        cdef RoomCluster cluster = self.room_clusters[k]

        cluster.update(goal, r)
        
    def get_mapping_probability(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef RoomCluster cluster = self.room_clusters[k]
        
        return cluster.get_mapping_probability(c, a, aa)

    def get_goal_probability(self, int c):
        cdef int k = self.room_assignments[c]
        cdef RoomCluster cluster = self.room_clusters[k]

        return cluster.get_goal_probability(c)
    
    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef RoomCluster cluster
        
        for cluster in self.room_clusters.values():
            log_likelihood += cluster.get_mapping_log_likelihood()
            log_likelihood += cluster.get_goal_log_likelihood()

        return log_likelihood

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.total_log_prior

    def get_log_prior(self):
        return self.total_log_prior
    
    def get_room_assignments(self):
        return self.room_assignments
    
    def get_assignments(self):
        cdef RoomCluster cluster
        cdef list goal_assignments = [], _goal_assignments, mapping_assignments = [], _mapping_assignments
        
        for cluster in self.room_clusters.values():
            _goal_assignments = [self.goal_assignments, cluster.get_goal_assignments()]
            _mapping_assignments = [self.mapping_assignments, cluster.get_mapping_assignments()]
            
            goal_assignments.append(_goal_assignments)
            mapping_assignments.append(_mapping_assignments)
            
        return goal_assignments, mapping_assignments
    
    def deep_copy(self):
        cdef RoomsHypothesis _h_copy = RoomsHypothesis(self.n_goals, self.n_primitive_actions,
                self.n_abstract_actions, self.alpha, self.room_prior, self.goal_prior, self.mapping_prior)

        _h_copy.goal_clusters = {k: cluster.deep_copy() for k, cluster in self.goal_clusters.iteritems()}
        _h_copy.mapping_clusters = {k: cluster.deep_copy() for k, cluster in self.mapping_clusters.iteritems()}
        _h_copy.room_clusters = {k: cluster.deep_copy(self.goal_clusters, _h_copy.goal_clusters, 
                        self.mapping_clusters, _h_copy.mapping_clusters) for k, cluster in self.room_clusters.iteritems()}

        _h_copy.room_assignments = {c: k for c,k in self.room_assignments.iteritems()}
        _h_copy.goal_assignments = {c: k for c,k in self.goal_assignments.iteritems()}
        _h_copy.mapping_assignments = {c: k for c,k in self.mapping_assignments.iteritems()}

        # initialize posterior
        _h_copy.room_log_prior = self.room_log_prior
        _h_copy.goal_log_prior = self.goal_log_prior
        _h_copy.mapping_log_prior = self.mapping_log_prior
        _h_copy.room_goal_mapping_log_prior = self.room_goal_mapping_log_prior
        _h_copy.total_log_prior = self.total_log_prior
    

cdef class RoomCluster(object):
    cdef int n_goals, n_primitive_actions, n_abstract_actions
    cdef double goal_log_prior, mapping_log_prior, total_log_prior, goal_prior, mapping_prior, alpha
    cdef dict goal_assignments, goal_clusters, mapping_assignments, mapping_clusters
    cdef list goal_experience, mapping_experience

    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions,
             double alpha, double goal_prior, double mapping_prior):

        self.n_goals = n_goals
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.goal_clusters = dict()
        self.mapping_clusters = dict()

        self.goal_assignments = dict()
        self.mapping_assignments = dict()

        self.alpha = alpha
        self.goal_prior = goal_prior
        self.mapping_prior = mapping_prior

        # initialize posterior
        self.goal_experience = list()
        self.mapping_experience = list()

        self.goal_log_prior = 1.0/2
        self.mapping_log_prior = 1.0/2

        self.total_log_prior = 1.0

    def add_new_goal_assignment(self, int c, int k, GoalCluster cluster):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.goal_clusters.keys():
            # if not, add an new reward cluster
            self.goal_clusters[k] = cluster

        self.goal_assignments[c] = k  # note, there's no check built in here
        self.goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha)

        self.total_log_prior = self.goal_log_prior + self.mapping_log_prior

    def add_new_mapping_assignment(self, int c, int k, MappingCluster cluster):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.mapping_assignments.values():
            # if not, add an new reward cluster
            self.mapping_clusters[k] = cluster

        self.mapping_assignments[c] = k
        self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha)
        
        self.total_log_prior = self.goal_log_prior + self.mapping_log_prior
        
    def get_room_goal_mapping_log_prior(self):
        return self.total_log_prior
    
    def update_mapping(self, int c, int a, int aa):
        cdef int k = self.mapping_assignments[c]
        cdef MappingCluster cluster = self.mapping_clusters[k]
        cluster.update(a, aa)

        # need to store all experiences for log probability calculations
        self.mapping_experience.append((k, a, aa))

    def update(self, int c, int goal, int r):
        cdef int k = self.goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]
        cluster.update(goal, r)

        self.goal_experience.append((k, goal, r))

    def get_mapping_probability(self, int c, int a, int aa):
        cdef int k = self.mapping_assignments[c]
        cdef MappingCluster cluster = self.mapping_clusters[k]
        return cluster.get_mapping_mle(a, aa)

    def get_goal_probability(self, int c):
        cdef int g
        cdef int k = self.goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]

        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)

        cdef double [:] rew_func = cluster.get_goal_pmf()
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability

    def get_mapping_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, a, aa
        cdef MappingCluster cluster

        #loop through experiences and get posterior
        for k, a, aa in self.mapping_experience:
            cluster = self.mapping_clusters[k]
            log_likelihood += log(cluster.get_likelihood(a, aa))
            
    def get_goal_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, goal, r
        cdef GoalCluster cluster

        #loop through experiences and get posterior
        for k, goal, r in self.goal_experience:
            cluster = self.goal_clusters[k]
            log_likelihood += log(cluster.get_observation_probability(goal, r))

        return log_likelihood
    
    def get_goal_assignments(self):
        return self.goal_assignments
    
    def get_mapping_assignments(self):
        return self.mapping_assignments
    
    def deep_copy(self, dict old_goal_clusters, dict new_goal_clusters, dict old_mapping_clusters, dict new_mapping_clusters):
        cdef int ii, jj
        cdef RoomCluster _cluster_copy = RoomCluster(self.n_goals, self.n_primitive_actions, 
                        self.n_abstract_actions, self.alpha, self.goal_prior, self.mapping_prior)

        _cluster_copy.goal_assignments = {c:k for c,k in self.goal_assignments.iteritems()}
        _cluster_copy.mapping_assignments = {c:k for c,k in self.mapping_assignments.iteritems()}

        _cluster_copy.mapping_experience = [(k, a, aa) for k, a, aa in self.mapping_experience]
        _cluster_copy.goal_experience = [(k, goal, r) for k, goal, r in self.goal_experience]

        _cluster_copy.goal_log_prior = self.goal_log_prior 
        _cluster_copy.mapping_log_prior = self.mapping_log_prior
        _cluster_copy.total_log_prior = self.total_log_prior

        _cluster_copy.goal_clusters = dict()
        for ii, cluster in self.goal_clusters.iteritems():
            for jj, cluster_old in old_goal_clusters.iteritems():
                if cluster == cluster_old:
                    _cluster_copy.goal_clusters[ii] = new_goal_clusters[jj]
                    break
        
        _cluster_copy.mapping_clusters = dict()
        for ii, cluster in self.mapping_clusters.iteritems():
            for jj, cluster_old in old_mapping_clusters.iteritems():
                if cluster == cluster_old:
                    _cluster_copy.mapping_clusters[ii] = new_mapping_clusters[jj]
                    break

        return _cluster_copy
