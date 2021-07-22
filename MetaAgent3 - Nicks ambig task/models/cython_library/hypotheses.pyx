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


cdef class MappingCluster:
    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a, log_pr_aa_given_a
    cdef double [:] abstract_action_counts, primitive_action_counts
    cdef int n_primitive_actions, n_abstract_actions
    cdef double mapping_prior

    def __cinit__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior, bint set_init):

        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a, log_pr_aa_given_a
        cdef double[:] abstract_action_counts, primitive_action_counts
        cdef int a, aa
        cdef double inv_n_a, inv_n_aa, mp_X_naa, mp_X_na

        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE)
        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=DTYPE)
        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=DTYPE)

        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE)
        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE)
        log_pr_aa_given_a = np.zeros((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE)

        if set_init:
            inv_n_a = 1.0 / n_primitive_actions
            inv_n_aa = 1.0 / n_abstract_actions
            mp_X_naa = mapping_prior * n_abstract_actions
            mp_X_na = mapping_prior * n_primitive_actions

            for a in range(n_primitive_actions):
                for aa in range(n_abstract_actions + 1):
                    mapping_history[a, aa] = mapping_prior
                    mapping_mle[a, aa] = inv_n_a
                    pr_aa_given_a[a, aa] = inv_n_aa

            for a in range(n_primitive_actions):
                primitive_action_counts[a] = mp_X_naa

            for aa in range(n_abstract_actions + 1):
                abstract_action_counts[aa] = mp_X_na

        self.mapping_history = mapping_history
        self.abstract_action_counts = abstract_action_counts
        self.mapping_mle = mapping_mle
        self.primitive_action_counts = primitive_action_counts
        self.pr_aa_given_a = pr_aa_given_a
        self.log_pr_aa_given_a = log_pr_aa_given_a

        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.mapping_prior = mapping_prior

    cdef update(self, int a, int aa):
        cdef int aa0, a0
        cdef float p, mh, aa_count

        self.mapping_history[a, aa] += 1.0
        self.abstract_action_counts[aa] += 1.0
        self.primitive_action_counts[a] += 1.0

        for aa0 in range(self.n_abstract_actions):
            aa_count = self.abstract_action_counts[aa0]

            for a0 in range(self.n_primitive_actions):
                mh = self.mapping_history[a0, aa0]
                self.mapping_mle[a0, aa0] =  mh / aa_count

                # p(A|a, k) estimator
                p = mh / self.primitive_action_counts[a0]
                self.pr_aa_given_a[a0, aa0] = p
                self.log_pr_aa_given_a[a0, aa0] = log(p)
                
    cpdef get_likelihood(self, int a, int aa):
        return self.pr_aa_given_a[a, aa]

    cpdef deep_copy(self):
        cdef int a, aa, idx, n_aa_w

        cdef MappingCluster _cluster_copy = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                                           self.mapping_prior, 0)

        n_aa_w = self.n_abstract_actions + 1 # include the possibility of the "wait" action

        # create local copy of the arrays before passing them whole to the new cluster
        cdef double[:, ::1] mapping_history = _cluster_copy.mapping_history
        cdef double[:, ::1] mapping_mle = _cluster_copy.mapping_mle
        cdef double[:, ::1] pr_aa_given_a = _cluster_copy.pr_aa_given_a
        cdef double[:, ::1] log_pr_aa_given_a = _cluster_copy.log_pr_aa_given_a
        cdef double[:] abstract_action_counts = _cluster_copy.abstract_action_counts
        cdef double[:] primitive_action_counts = _cluster_copy.primitive_action_counts

        for a in range(self.n_primitive_actions):

            primitive_action_counts[a] = self.primitive_action_counts[a]

            for aa in range(n_aa_w):
                mapping_history[a, aa] = self.mapping_history[a, aa]
                mapping_mle[a, aa] = self.mapping_mle[a, aa]
                pr_aa_given_a[a, aa] = self.pr_aa_given_a[a, aa]
                log_pr_aa_given_a[a, aa] = self.log_pr_aa_given_a[a, aa]

        for aa in range(n_aa_w):
            abstract_action_counts[aa] = self.abstract_action_counts[aa]

        _cluster_copy.primitive_action_counts = primitive_action_counts
        _cluster_copy.mapping_history = mapping_history
        _cluster_copy.mapping_mle = mapping_mle
        _cluster_copy.pr_aa_given_a = pr_aa_given_a
        _cluster_copy.log_pr_aa_given_a = log_pr_aa_given_a
        _cluster_copy.abstract_action_counts = abstract_action_counts

        return _cluster_copy


cdef class MappingHypothesis(object):

    cdef dict cluster_assignments, clusters
    cdef double prior_log_prob, alpha, mapping_prior
    cdef list experience_k
    cdef list experience_a
    cdef list experience_aa
    cdef int n_abstract_actions, n_primitive_actions, t
    cdef list visited_clusters

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
        self.prior_log_prob = 0.0

        # need to store all experiences for log probability calculations
        self.experience_k = list()
        self.experience_a = list()
        self.experience_aa = list()
        self.t = 0
        self.visited_clusters = []


    cpdef update_mapping(self, int c, int a, int aa):
        cdef int k = self.cluster_assignments[c]
        cdef MappingCluster cluster = self.clusters[k]

        cluster.update(a, aa)
        self.clusters[k] = cluster

        # need to store all experiences for log probability calculations
        self.experience_k.append(k)
        self.experience_a.append(a)
        self.experience_aa.append(aa)
        self.t += 1

        if k not in self.visited_clusters:
            self.visited_clusters.append(k)

    cpdef double get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef unsigned int k, k0, a, aa, t
        cdef MappingCluster cluster
        cdef double [:, ::1] ll_func

        #loop through experiences and get posterior
        for k in self.visited_clusters:

            # pre-cache cluster lookup b/c it is slow
            cluster = self.clusters[k]
            ll_func = cluster.log_pr_aa_given_a

            # now loop through and only pull the values for the current clusters
            t = 0
            while t < self.t:
                k0 = self.experience_k[t]
                if k == k0:
                    a = self.experience_a[t]
                    aa = self.experience_aa[t]
                    log_likelihood += ll_func[a, aa]
                t += 1

        return log_likelihood

    cpdef double get_log_posterior(self):
        return self.prior_log_prob + self.get_log_likelihood()

    cpdef get_mapping_probability(self, int c, int a, int aa):
        cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
        return cluster.mapping_mle[a, aa]

    cpdef get_log_prior(self):
        return self.prior_log_prob

    cpdef deep_copy(self):
        cdef MappingHypothesis _h_copy = MappingHypothesis(
            self.n_primitive_actions, self.n_abstract_actions, self.alpha,
            self.mapping_prior
        )

        cdef int k, c
        cdef MappingCluster cluster

        _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
        _h_copy.prior_log_prob = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)

        for k in self.visited_clusters:
            _h_copy.visited_clusters.append(k)
            cluster = self.clusters[k]
            _h_copy.clusters[k] = cluster.deep_copy()

        _h_copy.t = self.t
        _h_copy.experience_k = self.experience_k[:]
        _h_copy.experience_a = self.experience_a[:]
        _h_copy.experience_aa = self.experience_aa[:]

        return _h_copy

    cpdef get_assignments(self):
        return self.cluster_assignments

    cpdef add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior, 1)

        self.cluster_assignments[c] = k
        self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)

cdef class GoalCluster:
    cdef int n_goals
    cdef double set_visits, goal_prior
    cdef double [:]  goal_rewards_received, goal_reward_probability

    def __cinit__(self, int n_goals, float goal_prior, bint set_init):
        self.n_goals = n_goals
        self.goal_prior = goal_prior

        # rewards!
        cdef double [:] goal_rewards_received = np.ones(n_goals, dtype=DTYPE)
        cdef double [:] goal_reward_probability = np.ones(n_goals, dtype=DTYPE)

        cdef int g
        cdef double inv_n_g
        if set_init:
            inv_n_g = 1.0 / n_goals
            for g in range(n_goals):
                goal_rewards_received[g] = goal_prior
                goal_reward_probability[g] = inv_n_g

        self.set_visits =  n_goals * goal_prior
        self.goal_rewards_received = goal_rewards_received
        self.goal_reward_probability = goal_reward_probability

    cdef update(self, int goal, int r):
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


    cpdef double get_observation_probability(self, int goal, int r):
        if r == 0:
            return 1 - self.goal_reward_probability[goal]
        return self.goal_reward_probability[goal]


    cpdef deep_copy(self):
        cdef int g

        cdef GoalCluster _cluster_copy = GoalCluster(self.n_goals, self.goal_prior, 0)

        _cluster_copy.set_visits = self.set_visits

        cdef double [:] goal_rewards_received = _cluster_copy.goal_rewards_received
        cdef double [:] goal_reward_probability = _cluster_copy.goal_reward_probability

        for g in range(self.n_goals):
            goal_rewards_received[g] = self.goal_rewards_received[g]
            goal_reward_probability[g] = self.goal_reward_probability[g]

        _cluster_copy.goal_rewards_received = goal_rewards_received
        _cluster_copy.goal_reward_probability = goal_reward_probability

        return _cluster_copy


cdef class GoalHypothesis(object):
    cdef int n_goals, t
    cdef double log_prior, alpha, goal_prior
    cdef dict cluster_assignments, clusters
    cdef list experience_k
    cdef list experience_g
    cdef list experience_r

    def __init__(self, int n_goals, float alpha, float goal_prior):

        cdef dict cluster_assignments = dict()
        cdef dict clusters = dict()
        cdef list experience = list()

        self.n_goals = n_goals
        self.cluster_assignments = cluster_assignments
        self.alpha = alpha
        self.goal_prior = goal_prior

        # initialize goal clusters
        self.clusters = clusters

        # initialize posterior
        self.log_prior = 1.0

        self.experience_k = list()
        self.experience_g = list()
        self.experience_r = list()
        self.t = 0

    def update(self, int c, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        cluster.update(goal, r)
        self.clusters[k] = cluster

        self.experience_k.append(k)
        self.experience_g.append(goal)
        self.experience_r.append(r)
        self.t += 1

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, g, r
        cdef GoalCluster cluster

        #loop through experiences and get posterior
        for t in range(self.t):
            k = self.experience_k[t]
            g = self.experience_g[t]
            r = self.experience_r[t]
            cluster = self.clusters[k]
            log_likelihood += log(cluster.get_observation_probability(g, r))

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

        cdef double [:] rew_func = cluster.goal_reward_probability
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability


    def deep_copy(self):
        cdef GoalHypothesis _h_copy = GoalHypothesis(self.n_goals, self.alpha, self.goal_prior)

        cdef int c, k, t
        cdef GoalCluster cluster

        _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.log_prior = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)

        _h_copy.t = self.t
        _h_copy.experience_k = self.experience_k[:]
        _h_copy.experience_g = self.experience_g[:]
        _h_copy.experience_r = self.experience_r[:]

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = GoalCluster(self.n_goals, self.goal_prior, 1)

        self.cluster_assignments[c] = k  # note, there's no check built in here
        self.log_prior = get_prior_log_probability(self.cluster_assignments, self.alpha)

    def is_visited(self, int c):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        return cluster.set_visits >= 1
    
    
    
cdef class HierarchicalHypothesis(object):
    cdef int n_goals, n_primitive_actions, n_abstract_actions
    cdef double gamma, iteration_criterion, inverse_temperature, alpha0, alpha1
    cdef dict room_assignments, room_clusters, goal_assignments, goal_clusters, mapping_assignments, mapping_clusters
    cdef double room_log_prior, goal_log_prior, mapping_log_prior, goal_prior, mapping_prior


    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions, 
                 float inverse_temp, float gamma, float stop_criterion, float alpha0, 
                 float alpha1, float goal_prior, float mapping_prior):

        self.n_goals = n_goals
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
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
    def update(self, int c, int goal, int r):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        cluster.update(c, goal, r)
        
    def updating_mapping(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        cluster.updating_mapping(c, a, aa)
        
    def get_goal_probability(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.get_goal_probability(c)

    def get_mapping_probability(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.get_mapping_probability(c, a, aa)

    def get_log_posterior(self):
        cdef double log_posterior = self.room_log_prior + self.mapping_log_prior + self.goal_log_prior
        log_posterior += sum([cluster.get_goal_log_prior() + cluster.get_mapping_log_prior() + cluster.get_log_likelihood() for cluster in self.room_clusters.values()])
        return log_posterior
    
    def get_room_assignments(self):
        return self.room_assignments
    
    def get_goal_assignments(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]

        cdef list goal_assignments = [self.goal_assignments, cluster.get_goal_assignments()]
        
        return goal_assignments

    def get_mapping_assignments(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]

        cdef list mapping_assignments = [self.mapping_assignments, cluster.get_mapping_assignments()]
        
        return mapping_assignments

    def add_new_room_context_assignment(self, int c, int k):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.room_clusters.keys():
            # if not, add an new goal cluster
            self.room_clusters[k] = HierarchicalRoomCluster(self.n_goals, self.n_primitive_actions, 
                              self.n_abstract_actions, self.inverse_temperature, self.gamma, 
                              self.iteration_criterion, self.alpha1, 
                              self.goal_prior, self.mapping_prior)

        self.room_assignments[c] = k  # note, there's no check built in here
        self.room_log_prior = get_prior_log_probability(self.room_assignments, self.alpha0)
        
    def add_new_goal_context_assignment(self, int c, list hierarchical_assignment):
        cdef int k
        cdef GoalCluster cluster = None
        
        # check if new context has assignment in environment-wide CRP
        cdef dict env_assignments = hierarchical_assignment[0]
        if c in env_assignments.keys():
            k = env_assignments[c]
            
            # check if cluster "k" has already been assigned new cluster
            if k not in self.goal_clusters.keys():
                # if not, add an new goal cluster
                self.goal_clusters[k] = GoalCluster(self.n_goals, self.goal_prior, 1)

            self.goal_assignments[c] = k  # note, there's no check built in here
            self.goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha0)
            cluster = self.goal_clusters[k]
            
        k = self.room_assignments[c]
        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
        assert RoomCluster is not None
        
        k = hierarchical_assignment[1][c]
        RoomCluster.add_new_goal_context_assignment(c, k, cluster)
        
    def add_new_mapping_context_assignment(self, int c, list hierarchical_assignment):
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
                                                  self.mapping_prior, 1)
                
            self.mapping_assignments[c] = k  # note, there's no check built in here
            self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha0)
            cluster = self.mapping_clusters[k]
            
        k = self.room_assignments[c]
        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
        assert RoomCluster is not None
        
        k = hierarchical_assignment[1][c]
        RoomCluster.add_new_mapping_context_assignment(c, k, cluster)

    def deep_copy(self):
        cdef HierarchicalHypothesis _h_copy = HierarchicalHypothesis(self.n_goals, 
                self.n_primitive_actions, self.n_abstract_actions, self.inverse_temperature, 
                self.gamma, self.iteration_criterion, self.alpha0, self.alpha1, self.goal_prior, self.mapping_prior)

        _h_copy.room_log_prior = self.room_log_prior
        _h_copy.goal_log_prior = self.goal_log_prior
        _h_copy.mapping_log_prior = self.mapping_log_prior
        
        _h_copy.room_assignments = {c:k for c,k in self.room_assignments.iteritems()}
        _h_copy.goal_assignments = {c:k for c,k in self.goal_assignments.iteritems()}
        _h_copy.mapping_assignments = {c:k for c,k in self.mapping_assignments.iteritems()}
        
        _h_copy.goal_clusters = {k:cluster.deep_copy() for k,cluster in self.goal_clusters.iteritems()}
        _h_copy.mapping_clusters = {k:cluster.deep_copy() for k,cluster in self.mapping_clusters.iteritems()}
        
        _h_copy.room_clusters = {k:cluster.deep_copy(self.goal_clusters, _h_copy.goal_clusters, self.mapping_clusters, _h_copy.mapping_clusters) for k,cluster in self.room_clusters.iteritems()}

        return _h_copy


cdef class HierarchicalRoomCluster(object):
    cdef int n_goals, n_primitive_actions, n_abstract_actions
    cdef double gamma, iteration_criterion, inverse_temperature, alpha
    cdef dict room_assignments, room_clusters, goal_assignments, goal_clusters, mapping_assignments, mapping_clusters
    cdef list goal_experience, mapping_experience
    cdef double room_log_prior, goal_log_prior, mapping_log_prior, goal_prior, mapping_prior

    def __init__(self, int n_goals, int n_primitive_actions, int n_abstract_actions, 
                 float inverse_temp, float gamma, float stop_criterion, float alpha, 
                 float goal_prior, float mapping_prior):
        
        self.n_goals = n_goals
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.alpha = alpha
        
        self.goal_assignments = dict()
        self.mapping_assignments = dict()
        
        self.goal_clusters = dict()
        self.mapping_clusters = dict()

        self.goal_experience = list()        
        self.mapping_experience = list()

        # initialize posterior
        self.goal_log_prior = 0.
        self.mapping_log_prior = 0.

        self.goal_prior = goal_prior
        self.mapping_prior = mapping_prior
        
    # goal update
    def update(self, int c, int goal, int r):
        cdef int k = self.goal_assignments[c]
        cdef GoalCluster cluster = self.goal_clusters[k]
        cluster.update(goal, r)
        self.goal_experience.append((k, goal, r))
        
    def updating_mapping(self, int c, int a, int aa):
        cdef int k = self.mapping_assignments[c]
        cdef MappingCluster cluster = self.mapping_clusters[k]
        cluster.update(a, aa)
        self.mapping_experience.append((k, a, aa))
    
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

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, goal, r, aa
        cdef GoalCluster goal_cluster
        cdef MappingCluster mapping_cluster

        for k, goal, r in self.goal_experience:
            goal_cluster = self.goal_clusters[k]
            log_likelihood += log(goal_cluster.get_observation_probability(goal, r))

        #loop through experiences and get posterior
        for k, a, aa in self.mapping_experience:
            mapping_cluster = self.mapping_clusters[k]
            log_likelihood += log(mapping_cluster.get_likelihood(a, aa))

        return log_likelihood
    
    def get_goal_assignments(self):
        return self.goal_assignments
    
    def get_mapping_assignments(self):
        return self.mapping_assignments

    def add_new_goal_context_assignment(self, int c, int k, GoalCluster cluster):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.goal_clusters.keys():
            # if not, add an new goal cluster
            assert cluster is not None
            self.goal_clusters[k] = cluster

        self.goal_assignments[c] = k  # note, there's no check built in here
        self.goal_log_prior = get_prior_log_probability(self.goal_assignments, self.alpha)

    def add_new_mapping_context_assignment(self, int c, int k, MappingCluster cluster):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.mapping_clusters.keys():
            # if not, add an new goal cluster
            assert cluster is not None
            self.mapping_clusters[k] = cluster

        self.mapping_assignments[c] = k  # note, there's no check built in here
        self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha)
        
    def get_goal_log_prior(self):
        return self.goal_log_prior

    def get_mapping_log_prior(self):
        return self.mapping_log_prior
    
    def deep_copy(self, dict old_goal_clusters, dict new_goal_clusters, dict old_mapping_clusters, dict new_mapping_clusters):
        cdef int k, k2
        cdef GoalCluster goal_cluster, goal_cluster_old
        cdef MappingCluster mapping_cluster, mapping_cluster_old
        
        _cluster_copy = HierarchicalRoomCluster(self.n_goals, self.n_primitive_actions, 
                                                self.n_abstract_actions, self.inverse_temperature, 
                                                self.gamma, self.iteration_criterion, self.alpha, 
                                                self.goal_prior, self.mapping_prior)

        _cluster_copy.goal_log_prior = self.goal_log_prior 
        _cluster_copy.mapping_log_prior = self.mapping_log_prior

        _cluster_copy.goal_experience = [(k, goal, r) for k, goal, r in self.goal_experience]
        _cluster_copy.mapping_experience = [(k, a, aa) for k, a, aa in self.mapping_experience]

        _cluster_copy.goal_assignments = {c:k for c,k in self.goal_assignments.iteritems()}
        _cluster_copy.mapping_assignments = {c:k for c,k in self.mapping_assignments.iteritems()}
        
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
