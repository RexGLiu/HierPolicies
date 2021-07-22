#boundscheck=False, wraparound=True
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from core import value_iteration
from core import get_prior_log_probability

DTYPE = np.float
ctypedef np.float_t DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT_DTYPE_t

cdef extern from "math.h":
    double log(double x)
    
cdef extern from "math.h":
    double lgamma(double x)
    

from math import isnan


#cdef class MappingCluster(object):
#    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a
#    cdef double [:] abstract_action_counts, primitive_action_counts
#    cdef int n_primitive_actions, n_abstract_actions
#    cdef double mapping_prior
#    cdef list experience
#
#    def __init__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):
#
#        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a
#        cdef double[:] abstract_action_counts, primitive_action_counts
#
#        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=float) * mapping_prior
#        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=float) *  mapping_prior * n_primitive_actions
#        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=float) * \
#                      (1.0 / n_primitive_actions)
#
#        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE) * mapping_prior * n_abstract_actions
#        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE) * \
#                        (1.0 / n_abstract_actions)
#
#        self.mapping_history = mapping_history
#        self.abstract_action_counts = abstract_action_counts
#        self.mapping_mle = mapping_mle
#        self.primitive_action_counts = primitive_action_counts
#        self.pr_aa_given_a = pr_aa_given_a
#
#        self.n_primitive_actions = n_primitive_actions
#        self.n_abstract_actions = n_abstract_actions
#        self.mapping_prior = mapping_prior
#
#        # need to store all experiences for log probability calculations
#        self.experience = list()
#
#    def update(self, int a, int aa):
#        cdef int aa0, a0
#        self.mapping_history[a, aa] += 1.0
#        self.abstract_action_counts[aa] += 1.0
#        self.primitive_action_counts[a] += 1.0
#
#        for aa0 in range(self.n_abstract_actions):
#            for a0 in range(self.n_primitive_actions):
#                self.mapping_mle[a0, aa0] = self.mapping_history[a0, aa0] / self.abstract_action_counts[aa0]
#
#                # p(A|a, k) estimator
#                self.pr_aa_given_a[a0, aa0] = self.mapping_history[a0, aa0] / self.primitive_action_counts[a0]
#
#        # need to store all experiences for log probability calculations
#        self.experience.append((a, aa))
#        
#    def get_experience(self):
#        return self.experience
#
#    def get_mapping_mle(self, int a, int aa):
#        return self.mapping_mle[a, aa]
#
#    def get_likelihood(self, int a, int aa):
#        return self.pr_aa_given_a[a, aa]
#
#    def get_log_likelihood(self):
#        cdef double log_likelihood = 0
#        cdef int a, aa
#
#        #loop through experiences and get posterior
#        for a, aa in self.experience:
#            log_likelihood += log(self.get_likelihood(a, aa))
#            
##        print log_likelihood
#
#        return log_likelihood
#
#    def deep_copy(self):
#        cdef int a, aa, idx
#
#        cdef MappingCluster _cluster_copy = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
#                                                           self.mapping_prior)
#
#        _cluster_copy.experience = [(a, aa) for a, aa in self.experience]
#
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
#
#        return _cluster_copy
#
#cdef class MappingHypothesis(object):
#
#        cdef dict cluster_assignments, clusters
#        cdef double prior_log_prob, alpha, mapping_prior
#        cdef int n_abstract_actions, n_primitive_actions
#
#        def __init__(self, int n_primitive_actions, int n_abstract_actions, float alpha, float mapping_prior):
#
#            self.n_primitive_actions = n_primitive_actions
#            self.n_abstract_actions = n_abstract_actions
#            self.alpha = alpha
#            self.mapping_prior = mapping_prior
#
#            # initialize mapping clusters
#            self.clusters = dict()
#            self.cluster_assignments = dict()
#
#            # store the prior probability
#            self.prior_log_prob = 0
#
#        cdef _update_prior(self):
#            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)
#
#        cdef _get_cluster_average(self):
#            pass
#
#        def deep_copy(self):
#            cdef MappingHypothesis _h_copy = MappingHypothesis(self.n_primitive_actions, self.n_abstract_actions,
#                                                               self.alpha, self.mapping_prior)
#
#            cdef int k, a, aa, c
#            cdef MappingCluster cluster
#
#            # deep copy each list, dictionary, cluster, etc
#            _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
#            _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
#            _h_copy.prior_log_prob = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)
#            return _h_copy
#
#        def add_new_context_assignment(self, int c, int k):
#            """
#            :param c: context id number
#            :param k: cluster id number
#            :return:
#            """
#            # check if new cluster
#            if k not in self.cluster_assignments.values():
#                self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
#                                                  self.mapping_prior)
#
#            self.cluster_assignments[c] = k  # note, there's no check built in here
#            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)
#
#        def get_assignments(self):
#            return self.cluster_assignments
#
#        def updating_mapping(self, int c, int a, int aa):
#            cdef int k = self.cluster_assignments[c]
#            cdef MappingCluster cluster = self.clusters[k]
#            cluster.update(a, aa)
#            self.clusters[k] = cluster
#
#        def get_log_likelihood(self):
#            cdef double log_likelihood = 0
#            cdef MappingCluster cluster
#
#            #loop through experiences and get posterior
#            for k, cluster in self.clusters.iteritems():
#                log_likelihood += cluster.get_log_likelihood()
#                print k, cluster.get_experience()
#
#            print log_likelihood
#
#            return log_likelihood
#
#        def get_log_posterior(self):
#            return self.prior_log_prob + self.get_log_likelihood()
#
#        def get_mapping_probability(self, int c, int a, int aa):
#            cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
#            return cluster.get_mapping_mle(a, aa)
#
#        def get_log_prior(self):
#            return self.prior_log_prob





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

    def get_stats(self):
        return (self.pr_aa_given_a, self.mapping_history, self.abstract_action_counts, self.primitive_action_counts)

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

        def __init__(self, int n_primitive_actions, int n_abstract_actions, float alpha, float mapping_prior):

            self.n_primitive_actions = n_primitive_actions
            self.n_abstract_actions = n_abstract_actions
            self.alpha = alpha
            self.mapping_prior = mapping_prior

            # initialize mapping clusters
            self.clusters = dict()
            self.cluster_assignments = dict()

            # store the prior probability
            self.prior_log_prob = 0

            # need to store all experiences for log probability calculations
            self.experience = list()

        cdef _update_prior(self):
            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)

        cdef _get_cluster_average(self):
            pass

        def deep_copy(self):
            cdef MappingHypothesis _h_copy = MappingHypothesis(self.n_primitive_actions, self.n_abstract_actions,
                                                               self.alpha, self.mapping_prior)

            cdef int k, a, aa, c
            cdef MappingCluster cluster

            # deep copy each list, dictionary, cluster, etc
            _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
            _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
            _h_copy.experience = [(k, a, aa) for k, a, aa in self.experience]
            _h_copy.prior_log_prob = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)
            return _h_copy

        def add_new_context_assignment(self, int c, int k):
            """
            :param c: context id number
            :param k: cluster id number
            :return:
            """
            # check if new cluster
            if k not in self.cluster_assignments.values():
                self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                                  self.mapping_prior)

            self.cluster_assignments[c] = k  # note, there's no check built in here
            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)

        def get_assignments(self):
            return self.cluster_assignments

        def updating_mapping(self, int c, int a, int aa):
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
                
                
#            cdef double [:,::1] mapping_history, pr_aa_given_a
#            cdef double [:] abstract_action_counts, primitive_action_counts
#
#            for cluster in self.clusters.values():
#                (pr_aa_given_a, mapping_history, abstract_action_counts, primitive_action_counts) = cluster.get_stats()
#                print [list(pr_aa_given_a[a]) for a in range(self.n_primitive_actions)]
#                print [list(mapping_history[a]) for a in range(self.n_primitive_actions)]
#                print list(abstract_action_counts)
#                print list(primitive_action_counts)


#            print self.experience
            print "l", log_likelihood 

#            if len(self.experience) > 1 and self.experience[-1] == (3,0,3):
#                raise

            return log_likelihood

        def get_log_posterior(self):
            print "p", self.prior_log_prob
            return self.prior_log_prob + self.get_log_likelihood()

        def get_mapping_probability(self, int c, int a, int aa):
            cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
            return cluster.get_mapping_mle(a, aa)

        def get_log_prior(self):
            return self.prior_log_prob



#cdef class MappingCluster(object):
#    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a
#    cdef double [:] abstract_action_counts, primitive_action_counts
#    cdef int n_primitive_actions, n_abstract_actions
#    cdef double mapping_prior
#
#    def __init__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):
#
#        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a
#        cdef double[:] abstract_action_counts, primitive_action_counts
#
#        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=float) * mapping_prior
#        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=float) *  mapping_prior * n_primitive_actions
#        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=float) * \
#                      (1.0 / n_primitive_actions)
#
#        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE) * mapping_prior * n_abstract_actions
#        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE) * \
#                        (1.0 / n_abstract_actions)
#
#        self.mapping_history = mapping_history
#        self.abstract_action_counts = abstract_action_counts
#        self.mapping_mle = mapping_mle
#        self.primitive_action_counts = primitive_action_counts
#        self.pr_aa_given_a = pr_aa_given_a
#
#        self.n_primitive_actions = n_primitive_actions
#        self.n_abstract_actions = n_abstract_actions
#        self.mapping_prior = mapping_prior
#
#    def update(self, int a, int aa):
#        cdef int aa0, a0
#        self.mapping_history[a, aa] += 1.0
#        self.abstract_action_counts[aa] += 1.0
#        self.primitive_action_counts[a] += 1.0
#
#        for aa0 in range(self.n_abstract_actions):
#            for a0 in range(self.n_primitive_actions):
#                self.mapping_mle[a0, aa0] = self.mapping_history[a0, aa0] / self.abstract_action_counts[aa0]
#
#                # p(A|a, k) estimator
#                self.pr_aa_given_a[a0, aa0] = self.mapping_history[a0, aa0] / self.primitive_action_counts[a0]
#
#    def get_mapping_mle(self, int a, int aa):
#        return self.mapping_mle[a, aa]
#
#    def get_likelihood(self, int a, int aa):
#        return self.pr_aa_given_a[a, aa]
#    
#    def get_log_likelihood(self):
#        cdef double log_likelihood = 0.
#        cdef int a, aa
#        cdef double n
#        
#        for a in range(self.n_primitive_actions):
#            for aa in range(self.n_abstract_actions+1):
#                n = self.mapping_history[a, aa] - self.mapping_prior
#                log_likelihood += n*log(self.pr_aa_given_a[a, aa])
#                
#        print log_likelihood
#                
#        return log_likelihood
#
##    def get_log_likelihood(self):
##        cdef double log_likelihood = 0.
##        cdef int a, aa, n_aa
##        cdef double alpha, total, n
##        
##        n_aa = (self.n_abstract_actions+1)
##        
##        for a in range(self.n_primitive_actions):
##            total = self.primitive_action_counts[a]
##            for aa in range(n_aa):
##                alpha = self.mapping_history[a, aa]
##                log_likelihood += lgamma(alpha)
##            log_likelihood -= lgamma(total)
##        
##        n = self.n_primitive_actions - self.mapping_prior
##        log_likelihood -= n*(n_aa*lgamma(self.mapping_prior) - lgamma(n_aa*self.mapping_prior))
##
##        return log_likelihood
#
#    def deep_copy(self):
#        cdef int a, aa, idx
#
#        cdef MappingCluster _cluster_copy = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
#                                                           self.mapping_prior)
#
#        _cluster_copy.primitive_action_counts = np.array(self.primitive_action_counts)
#        _cluster_copy.mapping_history = np.array(self.mapping_history)
#        _cluster_copy.mapping_mle = np.array(self.mapping_mle)
#        _cluster_copy.pr_aa_given_a = np.array(self.pr_aa_given_a)
#        _cluster_copy.abstract_action_counts = np.array(self.abstract_action_counts)
#
#        return _cluster_copy
#
#cdef class MappingHypothesis(object):
#
#        cdef dict cluster_assignments, clusters, log_likelihood
#        cdef double prior_log_prob, alpha, mapping_prior
#        cdef int n_abstract_actions, n_primitive_actions
#
#        def __init__(self, int n_primitive_actions, int n_abstract_actions, float alpha, float mapping_prior):
#
#            self.n_primitive_actions = n_primitive_actions
#            self.n_abstract_actions = n_abstract_actions
#            self.alpha = alpha
#            self.mapping_prior = mapping_prior
#
#            # initialize mapping clusters
#            self.clusters = dict()
#            self.cluster_assignments = dict()
#            self.log_likelihood = dict()
#
#            # store the prior probability
#            self.prior_log_prob = 0
#
#        cdef _update_prior(self):
#            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)
#
#        cdef _get_cluster_average(self):
#            pass
#
#        def deep_copy(self):
#            cdef MappingHypothesis _h_copy = MappingHypothesis(self.n_primitive_actions, self.n_abstract_actions,
#                                                               self.alpha, self.mapping_prior)
#
#            cdef int k, a, aa, c
#            cdef MappingCluster cluster
#
#            # deep copy each list, dictionary, cluster, etc
#            _h_copy.cluster_assignments = dict(self.cluster_assignments)
#            _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
#            _h_copy.log_likelihood = dict(self.log_likelihood)
#            _h_copy.prior_log_prob = self.prior_log_prob
#            return _h_copy
#
#        def add_new_context_assignment(self, int c, int k):
#            """
#            :param c: context id number
#            :param k: cluster id number
#            :return:
#            """
#            # check if new cluster
#            if k not in self.cluster_assignments.values():
#                self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
#                                                  self.mapping_prior)
#                self.log_likelihood[k] = 0.
#
#            self.cluster_assignments[c] = k  # note, there's no check built in here
#            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)
#
#        def get_assignments(self):
#            return self.cluster_assignments
#
#        def updating_mapping(self, int c, int a, int aa):
#            cdef int k = self.cluster_assignments[c]
#            cdef MappingCluster cluster = self.clusters[k]
#            cluster.update(a, aa)
#            self.clusters[k] = cluster
#            self.log_likelihood[k] = cluster.get_log_likelihood()
#
#        def get_log_likelihood(self):
#            return sum(self.log_likelihood.values())
#
#        def get_log_posterior(self):
#            return self.prior_log_prob + self.get_log_likelihood()
#
#        def get_mapping_probability(self, int c, int a, int aa):
#            cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
#            return cluster.get_mapping_mle(a, aa)
#
#        def get_log_prior(self):
#            return self.prior_log_prob


cdef class RewardCluster(object):
    cdef double [:] reward_visits, reward_received, reward_function, reward_received_bool, log_likelihoods
    cdef double [:, ::1] reward_probability
    cdef double reward_prior

    def __init__(self, int n_stim):
        # rewards!
        self.reward_prior = 1e-2
        
        self.reward_visits = np.ones(n_stim) * self.reward_prior
        self.reward_received = np.ones(n_stim) * self.reward_prior
        self.reward_function = np.ones(n_stim) * 1.0

        # need a separate tracker for the probability a reward was received
        self.reward_received_bool = np.ones(n_stim) * self.reward_prior
        self.reward_probability   = np.ones((n_stim, 2)) * 0.5
        
        self.log_likelihoods = np.zeros(n_stim)

    def update(self, int sp, int r):
        self.reward_visits[sp] += 1.0
        self.reward_received[sp] += (r == 1.0)
        self.reward_function[sp] = self.reward_received[sp] / self.reward_visits[sp]

        self.reward_received_bool[sp] += float(r > 0)
        self.reward_probability[sp, 1] = self.reward_received_bool[sp] / self.reward_visits[sp]
        self.reward_probability[sp, 0] = 1 - self.reward_probability[sp, 1]
        
        # update log likelihoods
        cdef double n, n2, N
        N = self.reward_visits[sp] - self.reward_prior
        n = self.reward_received_bool[sp] - self.reward_prior
        n2 = N - n
        
        self.log_likelihoods[sp] = 0
        if n > 0:
            self.log_likelihoods[sp] += n*log(self.reward_probability[sp, 1])
        if n2 > 0:
            self.log_likelihoods[sp] += n2*log(self.reward_probability[sp, 0])

        
    def get_log_likelihood(self):
        return np.sum(self.log_likelihoods)

#    def get_log_likelihood(self):
#        cdef double log_likelihood = 0.
#        cdef int sp
#        cdef double alpha, beta, total
#        cdef int n_stim = len(self.reward_visits)
#        
#        for sp in range(n_stim):
#            total = self.reward_visits[sp]
#            alpha = self.reward_received[sp]
#            beta = total - alpha
#            log_likelihood += lgamma(alpha) + lgamma(beta) - lgamma(total)
#            
#        log_likelihood -= n_stim*(2*lgamma(self.goal_prior) - lgamma(2*self.goal_prior))
#        
#        return log_likelihood    

    def get_observation_probability(self, int sp, int r):
        cdef int idx = int(r>0)
        return self.reward_probability[sp, idx]

    def get_reward_prediction(self, int sp):
        if self.reward_visits[sp] > 0.1:
            return self.reward_function[sp]
        else:
            return 0

    def get_reward_function(self):
        return self.reward_function

    def get_reward_visits(self):
        return self.reward_function

#    def set_prior(self, list_goals):
#        cdef int s
#        cdef int n_stim = np.shape(self.reward_visits)[0]
#
#        # rewards!
#        self.reward_visits = np.ones(n_stim) * 0.0001
#        self.reward_received = np.ones(n_stim) * 0.00001
#
#        for s in list_goals:
#            self.reward_visits[s] += 0.001
#            self.reward_received[s] += 0.001
#
#        for s in range(n_stim):
#            self.reward_function[s] = self.reward_received[s] / self.reward_visits[s]

    def deep_copy(self):
        cdef int s, idx, n_stim

        n_stim = len(self.reward_visits)
        cdef RewardCluster _cluster_copy = RewardCluster(n_stim)

        _cluster_copy.reward_visits = np.array(self.reward_visits)
        _cluster_copy.reward_received = np.array(self.reward_received)
        _cluster_copy.reward_function = np.array(self.reward_function)
        _cluster_copy.reward_received_bool = np.array(self.reward_received_bool)
        _cluster_copy.reward_probability = np.array(self.reward_probability)
        _cluster_copy.log_likelihoods = np.array(self.log_likelihoods)

        return _cluster_copy


cdef class RewardHypothesis(object):
    cdef double gamma, iteration_criterion, log_prior, inverse_temperature, alpha
    cdef int n_stim
    cdef dict cluster_assignments, clusters
    cdef double [:,::1] reward_visits, reward_received, reward_function, reward_received_bool
    cdef double [:,:,::1] reward_probability

    def __init__(self, int n_stim, float inverse_temp, float gamma, float stop_criterion, float alpha):

        self.n_stim = n_stim
        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.cluster_assignments = dict()
        self.alpha = alpha

        # initialize mapping clusters
        self.clusters = {}

        # initialize posterior
        self.log_prior = 0

    def update(self, int c, int sp, int r):
        cdef int k = self.cluster_assignments[c]
        cdef RewardCluster cluster = self.clusters[k]
        cluster.update(sp, r)
        self.clusters[k] = cluster

    def deep_copy(self):
        cdef RewardHypothesis _h_copy = RewardHypothesis(self.n_stim, self.inverse_temperature, self.gamma,
                                                         self.iteration_criterion,  self.alpha)

        # deep copy each list, dictionary, cluster, etc.
        cdef int c, k, sp, r
        cdef RewardCluster cluster
        _h_copy.cluster_assignments = dict(self.cluster_assignments)
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.log_prior = self.log_prior

        return _h_copy

    def add_new_context_assignment(self, int c, int k):
        """
        :param c: context id number
        :param k: cluster id number
        :return:
        """
        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = RewardCluster(self.n_stim)

        self.cluster_assignments[c] = k  # note, there's no check built in here
        self.log_prior = get_prior_log_probability(self.cluster_assignments, self.alpha)

    def get_assignments(self):
        return self.cluster_assignments

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef RewardCluster cluster

        for cluster in self.clusters.values():
            log_likelihood += cluster.get_log_likelihood()

        return log_likelihood

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.log_prior

    def get_log_prior(self):
        return self.log_prior

    cpdef np.ndarray[DTYPE_t, ndim=1] get_abstract_action_q_values(self, int s, int c, double[:,:,::1] transition_function):
        cdef int k = self.cluster_assignments[c]
        cdef RewardCluster cluster = self.clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.asarray(cluster.get_reward_function())

        cdef double [:] v
        v = value_iteration(
            np.asarray(transition_function),
            reward_function,
            gamma=self.gamma,
            stop_criterion=self.iteration_criterion
        )

        cdef int n_abstract_actions = np.shape(transition_function)[1]

        # use the bellman equation to solve the q_values
        cdef np.ndarray q_values = np.zeros(n_abstract_actions)
        cdef int aa0, sp0
        for aa0 in range(n_abstract_actions):
            for sp0 in range(self.n_stim):
                q_values[aa0] += transition_function[s, aa0, sp0] * (reward_function[sp0] + self.gamma * v[sp0])

        return q_values

    def select_abstract_action_pmf(self, int s, int c, double[:,:,::1] transition_function):
        cdef np.ndarray[DTYPE_t, ndim=1] q_values = self.get_abstract_action_q_values(s, c, transition_function)

        # we need q-values to properly consider multiple options of equivalent optimality, but we can just always
        # pass a very high value for the temperature
        cdef np.ndarray[DTYPE_t, ndim=1] pmf = np.exp(np.array(q_values) * float(self.inverse_temperature))
        pmf = pmf / np.sum(pmf)

        return pmf

    def get_reward_function(self, int c):
        cdef int k = self.cluster_assignments[c]
        cdef RewardCluster cluster = self.clusters[k]

        cdef int n = len(cluster.get_reward_function())
        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.zeros(n, dtype=DTYPE)
        cdef int ii
        for ii in range(n):
            reward_function[ii] = cluster.get_reward_function()[ii]

        return reward_function

    def get_reward_visits(self, int c):
        cdef int k = self.cluster_assignments[c]
        cdef RewardCluster cluster = self.clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] reward_visits = np.asarray(cluster.get_reward_visits())

        return reward_visits

    def get_reward_prediction(self, int c, int sp):
        cdef int k = self.cluster_assignments[c]
        cdef RewardCluster cluster = self.clusters[k]
        cdef double r = cluster.get_reward_prediction(sp)
        return r

#    def set_reward_prior(self, list list_goals):
#        cdef int k
#        cdef RewardCluster cluster
#
#        for k in range(len(self.clusters)):
#            cluster = self.clusters[k]
#            cluster.set_prior(list_goals)


cdef class HierarchicalHypothesis(object):
    cdef int n_stim, n_primitive_actions, n_abstract_actions
    cdef double gamma, iteration_criterion, inverse_temperature, alpha0, alpha1
    cdef dict room_assignments, room_clusters, reward_assignments, reward_clusters, mapping_assignments, mapping_clusters
    cdef double room_log_prior, reward_log_prior, mapping_log_prior, mapping_prior


    def __init__(self, int n_stim, int n_primitive_actions, int n_abstract_actions, 
                 float inverse_temp, float gamma, float stop_criterion, float alpha0, 
                 float alpha1, float mapping_prior):

        self.n_stim = n_stim
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
        self.room_assignments = dict()
        self.reward_assignments = dict()
        self.mapping_assignments = dict()
        
        self.room_clusters = dict()
        self.reward_clusters = dict()
        self.mapping_clusters = dict()

        # initialize posterior
        self.room_log_prior = 0.
        self.reward_log_prior = 0.
        self.mapping_log_prior = 0.

        self.mapping_prior = mapping_prior
    
    # reward update
    def update(self, int c, int sp, int r):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        cluster.update(c, sp, r)
        
    def updating_mapping(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        cluster.updating_mapping(c, a, aa)
        
    def select_abstract_action_pmf(self, int s, int c, double[:,:,::1] transition_function):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.select_abstract_action_pmf(s, c, transition_function)

    def get_reward_function(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.get_reward_function(c)

    def get_reward_prediction(self, int c, int sp):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.get_reward_prediction(c,sp)

#    def set_reward_prior(self, list list_goals):
#        cdef RewardCluster cluster
#
#        for cluster in self.reward_clusters.values():
#            cluster.set_prior(list_goals)

    def get_mapping_probability(self, int c, int a, int aa):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
        return cluster.get_mapping_probability(c, a, aa)

    cpdef double get_log_posterior(self):
        cdef double log_posterior = self.room_log_prior + self.mapping_log_prior + self.reward_log_prior
        log_posterior += sum([cluster.get_reward_log_prior() + cluster.get_mapping_log_prior() + cluster.get_log_likelihood() for cluster in self.room_clusters.values()])
        return log_posterior
    
    def get_room_assignments(self):
        return self.room_assignments
    
    def get_reward_assignments(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]

        cdef list reward_assignments = [self.reward_assignments, cluster.get_reward_assignments()]
        
        return reward_assignments

    def get_mapping_assignments(self, int c):
        cdef int k = self.room_assignments[c]
        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]

        cdef list mapping_assignments = [self.mapping_assignments, cluster.get_mapping_assignments()]
        
        return mapping_assignments

    def add_new_room_context_assignment(self, int c, int k):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.room_clusters.keys():
            # if not, add an new reward cluster
            self.room_clusters[k] = HierarchicalRoomCluster(self.n_stim, self.n_primitive_actions, 
                              self.n_abstract_actions, self.inverse_temperature, self.gamma, 
                              self.iteration_criterion, self.alpha0, self.alpha1, self.mapping_prior)

        self.room_assignments[c] = k  # note, there's no check built in here
        self.room_log_prior = get_prior_log_probability(self.room_assignments, self.alpha1)
        
    def add_new_reward_context_assignment(self, int c, list hierarchical_assignment):
        cdef int k
        cdef RewardCluster cluster = None
        
        # check if new context has assignment in environment-wide CRP
        cdef dict env_assignments = hierarchical_assignment[0]
        if c in env_assignments.keys():
            k = env_assignments[c]
            
            # check if cluster "k" has already been assigned new cluster
            if k not in self.reward_clusters.keys():
                # if not, add an new reward cluster
                self.reward_clusters[k] = RewardCluster(self.n_stim)

            self.reward_assignments[c] = k  # note, there's no check built in here
            self.reward_log_prior = get_prior_log_probability(self.reward_assignments, self.alpha0)
            cluster = self.reward_clusters[k]
            
        k = self.room_assignments[c]
        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
        assert RoomCluster is not None
        
        k = hierarchical_assignment[1][c]
        RoomCluster.add_new_reward_context_assignment(c, k, cluster)
        
    def add_new_mapping_context_assignment(self, int c, list hierarchical_assignment):
        cdef int k
        cdef MappingCluster cluster = None
        
        # check if new context has assignment in environment-wide CRP
        cdef dict env_assignments = hierarchical_assignment[0]
        if c in env_assignments.keys():
            k = env_assignments[c]
            
            # check if cluster "k" has already been assigned new cluster
            if k not in self.mapping_clusters.keys():
                # if not, add an new reward cluster
                self.mapping_clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                                  self.mapping_prior)
                
            self.mapping_assignments[c] = k  # note, there's no check built in here
            self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha0)
            cluster = self.mapping_clusters[k]
            
        k = self.room_assignments[c]
        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
        assert RoomCluster is not None
        
        k = hierarchical_assignment[1][c]
        RoomCluster.add_new_mapping_context_assignment(c, k, cluster)

    def deep_copy(self):
        cdef HierarchicalHypothesis _h_copy = HierarchicalHypothesis(self.n_stim, 
                self.n_primitive_actions, self.n_abstract_actions, self.inverse_temperature, 
                self.gamma, self.iteration_criterion, self.alpha0, self.alpha1, self.mapping_prior)

        _h_copy.room_log_prior = self.room_log_prior
        _h_copy.reward_log_prior = self.reward_log_prior
        _h_copy.mapping_log_prior = self.mapping_log_prior
        
        _h_copy.room_assignments = dict(self.room_assignments)
        _h_copy.reward_assignments = dict(self.reward_assignments)
        _h_copy.mapping_assignments = dict(self.mapping_assignments)
        
        _h_copy.reward_clusters = {k:cluster.deep_copy() for k,cluster in self.reward_clusters.iteritems()}
        _h_copy.mapping_clusters = {k:cluster.deep_copy() for k,cluster in self.mapping_clusters.iteritems()}
        
        _h_copy.room_clusters = {k:cluster.deep_copy(self.reward_clusters, _h_copy.reward_clusters, self.mapping_clusters, _h_copy.mapping_clusters) for k,cluster in self.room_clusters.iteritems()}

        return _h_copy


cdef class HierarchicalRoomCluster(object):
    cdef int n_stim, n_primitive_actions, n_abstract_actions
    cdef double gamma, iteration_criterion, inverse_temperature, alpha0, alpha1
    cdef dict room_assignments, room_clusters, reward_assignments, reward_clusters, mapping_assignments, mapping_clusters
    cdef list reward_experience, mapping_experience
    cdef double room_log_prior, reward_log_prior, mapping_log_prior, mapping_prior

    def __init__(self, int n_stim, int n_primitive_actions, int n_abstract_actions, 
                 float inverse_temp, float gamma, float stop_criterion, float alpha0, 
                 float alpha1, float mapping_prior):
        
        self.n_stim = n_stim
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions

        self.inverse_temperature = inverse_temp
        self.gamma = gamma
        self.iteration_criterion = stop_criterion
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        
        self.reward_assignments = dict()
        self.mapping_assignments = dict()
        
        self.reward_clusters = dict()
        self.mapping_clusters = dict()

        self.reward_experience = list()        
        self.mapping_experience = list()        

        # initialize posterior
        self.reward_log_prior = 0.
        self.mapping_log_prior = 0.

        self.mapping_prior = mapping_prior
        
        
    # reward update
    def update(self, int c, int sp, int r):
        cdef int k = self.reward_assignments[c]
        cdef RewardCluster cluster = self.reward_clusters[k]
        cluster.update(sp, r)
        self.reward_experience.append((k, sp, r))
        
        
    def updating_mapping(self, int c, int a, int aa):
        cdef int k = self.mapping_assignments[c]
        cdef MappingCluster cluster = self.mapping_clusters[k]

        cluster.update(a, aa)
        self.mapping_experience.append((k, a, aa))
        
        
    cpdef np.ndarray[DTYPE_t, ndim=1] get_abstract_action_q_values(self, int s, int c, double[:,:,::1] transition_function):
        cdef int k = self.reward_assignments[c]
        cdef RewardCluster cluster = self.reward_clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.asarray(cluster.get_reward_function())

        cdef double [:] v
        v = value_iteration(
            np.asarray(transition_function),
            reward_function,
            gamma=self.gamma,
            stop_criterion=self.iteration_criterion
        )

        cdef int n_abstract_actions = np.shape(transition_function)[1]

        # use the bellman equation to solve the q_values
        cdef np.ndarray q_values = np.zeros(n_abstract_actions)
        cdef int aa0, sp0
        for aa0 in range(n_abstract_actions):
            for sp0 in range(self.n_stim):
                q_values[aa0] += transition_function[s, aa0, sp0] * (reward_function[sp0] + self.gamma * v[sp0])

        return q_values
    
    cpdef np.ndarray[DTYPE_t, ndim=1] select_abstract_action_pmf(self, int s, int c, double[:,:,::1] transition_function):
        cdef np.ndarray[DTYPE_t, ndim=1] q_values = self.get_abstract_action_q_values(s, c, transition_function)

        # we need q-values to properly consider multiple options of equivalent optimality, but we can just always
        # pass a very high value for the temperature
        cdef np.ndarray[DTYPE_t, ndim=1] pmf = np.exp(np.array(q_values) * float(self.inverse_temperature))
        pmf = pmf / np.sum(pmf)

        return pmf

    cpdef np.ndarray[DTYPE_t, ndim=1] get_reward_function(self, int c):
        cdef int k = self.reward_assignments[c]
        cdef RewardCluster cluster = self.reward_clusters[k]

        cdef int n = len(cluster.get_reward_function())
        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.zeros(n, dtype=DTYPE)
        cdef int ii
        for ii in range(n):
            reward_function[ii] = cluster.get_reward_function()[ii]

        return reward_function

    cpdef double get_reward_prediction(self, int c, int sp):
        cdef int k = self.reward_assignments[c]
        cdef RewardCluster cluster = self.reward_clusters[k]
        cdef double r = cluster.get_reward_prediction(sp)
        return r

    def get_mapping_clusters(self):
        return self.mapping_clusters

    def get_reward_clusters(self):
        return self.reward_clusters

    def get_mapping_probability(self, int c, int a, int aa):
        cdef MappingCluster cluster = self.mapping_clusters[self.mapping_assignments[c]]
        return cluster.get_mapping_mle(a, aa)

    cpdef double get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, sp, r, aa
        cdef RewardCluster reward_cluster
        cdef MappingCluster mapping_cluster

        for k, sp, r in self.reward_experience:
            reward_cluster = self.reward_clusters[k]
            log_likelihood += log(reward_cluster.get_observation_probability(sp, r))

        #loop through experiences and get posterior
        for k, a, aa in self.mapping_experience:
            mapping_cluster = self.mapping_clusters[k]
            log_likelihood += log(mapping_cluster.get_likelihood(a, aa))

        return log_likelihood
    
    def get_reward_assignments(self):
        return self.reward_assignments
    
    def get_mapping_assignments(self):
        return self.mapping_assignments

    def add_new_reward_context_assignment(self, int c, int k, RewardCluster cluster):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.reward_clusters.keys():
            # if not, add an new reward cluster
            assert cluster is not None
            self.reward_clusters[k] = cluster

        self.reward_assignments[c] = k  # note, there's no check built in here
        self.reward_log_prior = get_prior_log_probability(self.reward_assignments, self.alpha1)

    def add_new_mapping_context_assignment(self, int c, int k, MappingCluster cluster):
        # check if cluster "k" is already been assigned new cluster
        if k not in self.mapping_clusters.keys():
            # if not, add an new reward cluster
            assert cluster is not None
            self.mapping_clusters[k] = cluster

        self.mapping_assignments[c] = k  # note, there's no check built in here
        self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha1)
        
    cpdef double get_reward_log_prior(self):
        return self.reward_log_prior

    cpdef double get_mapping_log_prior(self):
        return self.mapping_log_prior
    
    def deep_copy(self, dict old_reward_clusters, dict new_reward_clusters, dict old_mapping_clusters, dict new_mapping_clusters):
        cdef int k, k2
        cdef RewardCluster reward_cluster, reward_cluster_old
        cdef MappingCluster mapping_cluster, mapping_cluster_old
        
        _cluster_copy = HierarchicalRoomCluster(self.n_stim, self.n_primitive_actions, 
                                                self.n_abstract_actions, self.inverse_temperature, 
                                                self.gamma, self.iteration_criterion, self.alpha0, 
                                                self.alpha1, self.mapping_prior)

        _cluster_copy.reward_log_prior = self.reward_log_prior 
        _cluster_copy.mapping_log_prior = self.mapping_log_prior

        _cluster_copy.reward_experience = [(k, sp, r) for k, sp, r in self.reward_experience]
        _cluster_copy.mapping_experience = [(k, a, aa) for k, a, aa in self.mapping_experience]

        _cluster_copy.reward_assignments = dict(self.reward_assignments)
        _cluster_copy.mapping_assignments = dict(self.mapping_assignments)
        
        _cluster_copy.reward_clusters = dict()
        for k, reward_cluster in self.reward_clusters.iteritems():
            for k2, reward_cluster_old in old_reward_clusters.iteritems():
                if reward_cluster == reward_cluster_old:
                    _cluster_copy.reward_clusters[k] = new_reward_clusters[k2]
                    break

        _cluster_copy.mapping_clusters = dict()
        for k, mapping_cluster in self.mapping_clusters.iteritems():
            for k2, mapping_cluster_old in old_mapping_clusters.iteritems():
                if mapping_cluster == mapping_cluster_old:
                    _cluster_copy.mapping_clusters[k] = new_mapping_clusters[k2]
                    break

        return _cluster_copy
    
    
    
    
    
    
    
    
    
    
    
##boundscheck=False, wraparound=True
#from __future__ import division
#import numpy as np
#cimport numpy as np
#cimport cython
#
#from core import value_iteration
#from core import get_prior_log_probability
#
#DTYPE = np.float
#ctypedef np.float_t DTYPE_t
#
#INT_DTYPE = np.int32
#ctypedef np.int32_t INT_DTYPE_t
#
#cdef extern from "math.h":
#    double log(double x)
#    
#cdef extern from "math.h":
#    double lgamma(double x)
#
#
#
#cdef class MappingCluster(object):
#    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a
#    cdef double [:] abstract_action_counts, primitive_action_counts
#    cdef int n_primitive_actions, n_abstract_actions
#    cdef double mapping_prior
#
#    def __init__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):
#
#        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a
#        cdef double[:] abstract_action_counts, primitive_action_counts
#
#        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=float) * mapping_prior
#        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=float) *  mapping_prior * n_primitive_actions
#        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=float) * \
#                      (1.0 / n_primitive_actions)
#
#        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE) * mapping_prior * n_abstract_actions
#        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE) * \
#                        (1.0 / n_abstract_actions)
#
#        self.mapping_history = mapping_history
#        self.abstract_action_counts = abstract_action_counts
#        self.mapping_mle = mapping_mle
#        self.primitive_action_counts = primitive_action_counts
#        self.pr_aa_given_a = pr_aa_given_a
#
#        self.n_primitive_actions = n_primitive_actions
#        self.n_abstract_actions = n_abstract_actions
#        self.mapping_prior = mapping_prior
#
#    def update(self, int a, int aa):
#        cdef int aa0, a0
#        self.mapping_history[a, aa] += 1.0
#        self.abstract_action_counts[aa] += 1.0
#        self.primitive_action_counts[a] += 1.0
#
#        for aa0 in range(self.n_abstract_actions):
#            for a0 in range(self.n_primitive_actions):
#                self.mapping_mle[a0, aa0] = self.mapping_history[a0, aa0] / self.abstract_action_counts[aa0]
#
#                # p(A|a, k) estimator
#                self.pr_aa_given_a[a0, aa0] = self.mapping_history[a0, aa0] / self.primitive_action_counts[a0]
#
#    def get_mapping_mle(self, int a, int aa):
#        return self.mapping_mle[a, aa]
#
#    def get_likelihood(self, int a, int aa):
#        return self.pr_aa_given_a[a, aa]
#    
#    def get_log_likelihood(self):
#        cdef double log_likelihood = 0.
#        cdef int a, aa
#        cdef double n
#        
#        for a in range(self.n_primitive_actions):
#            for aa in range(self.n_abstract_actions+1):
#                n = self.mapping_history[a, aa] - self.mapping_prior
#                log_likelihood += n*log(self.pr_aa_given_a[a, aa])
#                
#        return log_likelihood
#
##    def get_log_likelihood(self):
##        cdef double log_likelihood = 0.
##        cdef int a, aa, n_aa
##        cdef double alpha, total, n
##        
##        n_aa = (self.n_abstract_actions+1)
##        
##        for a in range(self.n_primitive_actions):
##            total = self.primitive_action_counts[a]
##            for aa in range(n_aa):
##                alpha = self.mapping_history[a, aa]
##                log_likelihood += lgamma(alpha)
##            log_likelihood -= lgamma(total)
##        
##        n = self.n_primitive_actions - self.mapping_prior
##        log_likelihood -= n*(n_aa*lgamma(self.mapping_prior) - lgamma(n_aa*self.mapping_prior))
##
##        return log_likelihood
#
#    def deep_copy(self):
#        cdef int a, aa, idx
#
#        cdef MappingCluster _cluster_copy = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
#                                                           self.mapping_prior)
#
#        _cluster_copy.primitive_action_counts = np.array(self.primitive_action_counts)
#        _cluster_copy.mapping_history = np.array(self.mapping_history)
#        _cluster_copy.mapping_mle = np.array(self.mapping_mle)
#        _cluster_copy.pr_aa_given_a = np.array(self.pr_aa_given_a)
#        _cluster_copy.abstract_action_counts = np.array(self.abstract_action_counts)
#
#        return _cluster_copy
#
#cdef class MappingHypothesis(object):
#
#        cdef dict cluster_assignments, clusters, log_likelihood
#        cdef double prior_log_prob, alpha, mapping_prior
#        cdef int n_abstract_actions, n_primitive_actions
#
#        def __init__(self, int n_primitive_actions, int n_abstract_actions, float alpha, float mapping_prior):
#
#            self.n_primitive_actions = n_primitive_actions
#            self.n_abstract_actions = n_abstract_actions
#            self.alpha = alpha
#            self.mapping_prior = mapping_prior
#
#            # initialize mapping clusters
#            self.clusters = dict()
#            self.cluster_assignments = dict()
#            self.log_likelihood = dict()
#
#            # store the prior probability
#            self.prior_log_prob = 0
#
#        cdef _update_prior(self):
#            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)
#
#        cdef _get_cluster_average(self):
#            pass
#
#        def deep_copy(self):
#            cdef MappingHypothesis _h_copy = MappingHypothesis(self.n_primitive_actions, self.n_abstract_actions,
#                                                               self.alpha, self.mapping_prior)
#
#            cdef int k, a, aa, c
#            cdef MappingCluster cluster
#
#            # deep copy each list, dictionary, cluster, etc
#            _h_copy.cluster_assignments = dict(self.cluster_assignments)
#            _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
#            _h_copy.log_likelihood = dict(self.log_likelihood)
#            _h_copy.prior_log_prob = self.prior_log_prob
#            return _h_copy
#
#        def add_new_context_assignment(self, int c, int k):
#            """
#            :param c: context id number
#            :param k: cluster id number
#            :return:
#            """
#            # check if new cluster
#            if k not in self.cluster_assignments.values():
#                self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
#                                                  self.mapping_prior)
#                self.log_likelihood[k] = 0.
#
#            self.cluster_assignments[c] = k  # note, there's no check built in here
#            self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)
#
#        def get_assignments(self):
#            return self.cluster_assignments
#
#        def updating_mapping(self, int c, int a, int aa):
#            cdef int k = self.cluster_assignments[c]
#            cdef MappingCluster cluster = self.clusters[k]
#            cluster.update(a, aa)
#            self.clusters[k] = cluster
#            self.log_likelihood[k] = cluster.get_log_likelihood()
#
#        def get_log_likelihood(self):
#            return sum(self.log_likelihood.values())
#
#        def get_log_posterior(self):
#            return self.prior_log_prob + self.get_log_likelihood()
#
#        def get_mapping_probability(self, int c, int a, int aa):
#            cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
#            return cluster.get_mapping_mle(a, aa)
#
#        def get_log_prior(self):
#            return self.prior_log_prob
#
#
#cdef class RewardCluster(object):
#    cdef double [:] reward_visits, reward_received, reward_function, reward_received_bool, log_likelihoods
#    cdef double [:, ::1] reward_probability
#    cdef double reward_prior
#
#    def __init__(self, int n_stim):
#        # rewards!
#        self.reward_prior = 1e-2
#        
#        self.reward_visits = np.ones(n_stim) * self.reward_prior
#        self.reward_received = np.ones(n_stim) * self.reward_prior
#        self.reward_function = np.ones(n_stim) * 1.0
#
#        # need a separate tracker for the probability a reward was received
#        self.reward_received_bool = np.ones(n_stim) * self.reward_prior
#        self.reward_probability   = np.ones((n_stim, 2)) * 0.5
#        
#        self.log_likelihoods = np.zeros(n_stim)
#
#    def update(self, int sp, int r):
#        self.reward_visits[sp] += 1.0
#        self.reward_received[sp] += (r == 1.0)
#        self.reward_function[sp] = self.reward_received[sp] / self.reward_visits[sp]
#
#        self.reward_received_bool[sp] += float(r > 0)
#        self.reward_probability[sp, 1] = self.reward_received_bool[sp] / self.reward_visits[sp]
#        self.reward_probability[sp, 0] = 1 - self.reward_probability[sp, 1]
#        
#        # update log likelihoods
#        cdef double n, n2, N
#        N = self.reward_visits[sp] - self.reward_prior
#        n = self.reward_received_bool[sp] - self.reward_prior
#        n2 = N - n
#        
#        self.log_likelihoods[sp] = 0
#        if n > 0:
#            self.log_likelihoods[sp] += n*log(self.reward_probability[sp, 1])
#        if n2 > 0:
#            self.log_likelihoods[sp] += n2*log(self.reward_probability[sp, 0])
#
#        
#    def get_log_likelihood(self):
#        return np.sum(self.log_likelihoods)
#
##    def get_log_likelihood(self):
##        cdef double log_likelihood = 0.
##        cdef int sp
##        cdef double alpha, beta, total
##        cdef int n_stim = len(self.reward_visits)
##        
##        for sp in range(n_stim):
##            total = self.reward_visits[sp]
##            alpha = self.reward_received[sp]
##            beta = total - alpha
##            log_likelihood += lgamma(alpha) + lgamma(beta) - lgamma(total)
##            
##        log_likelihood -= n_stim*(2*lgamma(self.goal_prior) - lgamma(2*self.goal_prior))
##        
##        return log_likelihood    
#
#    def get_observation_probability(self, int sp, int r):
#        cdef int idx = int(r>0)
#        return self.reward_probability[sp, idx]
#
#    def get_reward_prediction(self, int sp):
#        if self.reward_visits[sp] > 0.1:
#            return self.reward_function[sp]
#        else:
#            return 0
#
#    def get_reward_function(self):
#        return self.reward_function
#
#    def get_reward_visits(self):
#        return self.reward_function
#
#    def set_prior(self, list_goals):
#        cdef int s
#        cdef int n_stim = np.shape(self.reward_visits)[0]
#
#        # rewards!
#        self.reward_visits = np.ones(n_stim) * 0.0001
#        self.reward_received = np.ones(n_stim) * 0.00001
#
#        for s in list_goals:
#            self.reward_visits[s] += 0.001
#            self.reward_received[s] += 0.001
#
#        for s in range(n_stim):
#            self.reward_function[s] = self.reward_received[s] / self.reward_visits[s]
#
#    def deep_copy(self):
#        cdef int s, idx, n_stim
#
#        n_stim = len(self.reward_visits)
#        cdef RewardCluster _cluster_copy = RewardCluster(n_stim)
#
#        _cluster_copy.reward_visits = np.array(self.reward_visits)
#        _cluster_copy.reward_received = np.array(self.reward_received)
#        _cluster_copy.reward_function = np.array(self.reward_function)
#        _cluster_copy.reward_received_bool = np.array(self.reward_received_bool)
#        _cluster_copy.reward_probability = np.array(self.reward_probability)
#        _cluster_copy.log_likelihoods = np.array(self.log_likelihoods)
#
#        return _cluster_copy
#
#
#cdef class RewardHypothesis(object):
#    cdef double gamma, iteration_criterion, log_prior, inverse_temperature, alpha
#    cdef int n_stim
#    cdef dict cluster_assignments, clusters
#    cdef double [:,::1] reward_visits, reward_received, reward_function, reward_received_bool
#    cdef double [:,:,::1] reward_probability
#
#    def __init__(self, int n_stim, float inverse_temp, float gamma, float stop_criterion, float alpha):
#
#        self.n_stim = n_stim
#        self.inverse_temperature = inverse_temp
#        self.gamma = gamma
#        self.iteration_criterion = stop_criterion
#        self.cluster_assignments = dict()
#        self.alpha = alpha
#
#        # initialize mapping clusters
#        self.clusters = {}
#
#        # initialize posterior
#        self.log_prior = 0
#
#    def update(self, int c, int sp, int r):
#        cdef int k = self.cluster_assignments[c]
#        cdef RewardCluster cluster = self.clusters[k]
#        cluster.update(sp, r)
#        self.clusters[k] = cluster
#
#    def deep_copy(self):
#        cdef RewardHypothesis _h_copy = RewardHypothesis(self.n_stim, self.inverse_temperature, self.gamma,
#                                                         self.iteration_criterion,  self.alpha)
#
#        # deep copy each list, dictionary, cluster, etc.
#        cdef int c, k, sp, r
#        cdef RewardCluster cluster
#        _h_copy.cluster_assignments = dict(self.cluster_assignments)
#        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
#        _h_copy.log_prior = self.log_prior
#
#        return _h_copy
#
#    def add_new_context_assignment(self, int c, int k):
#        """
#        :param c: context id number
#        :param k: cluster id number
#        :return:
#        """
#        # check if cluster "k" is already been assigned new cluster
#        if k not in self.cluster_assignments.values():
#            # if not, add an new reward cluster
#            self.clusters[k] = RewardCluster(self.n_stim)
#
#        self.cluster_assignments[c] = k  # note, there's no check built in here
#        self.log_prior = get_prior_log_probability(self.cluster_assignments, self.alpha)
#
#    def get_assignments(self):
#        return self.cluster_assignments
#
#    def get_log_likelihood(self):
#        cdef double log_likelihood = 0
#        cdef RewardCluster cluster
#
#        for cluster in self.clusters.values():
#            log_likelihood += cluster.get_log_likelihood()
#
#        return log_likelihood
#
#    def get_log_posterior(self):
#        return self.get_log_likelihood() + self.log_prior
#
#    def get_log_prior(self):
#        return self.log_prior
#
#    cpdef np.ndarray[DTYPE_t, ndim=1] get_abstract_action_q_values(self, int s, int c, double[:,:,::1] transition_function):
#        cdef int k = self.cluster_assignments[c]
#        cdef RewardCluster cluster = self.clusters[k]
#        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.asarray(cluster.get_reward_function())
#
#        cdef double [:] v
#        v = value_iteration(
#            np.asarray(transition_function),
#            reward_function,
#            gamma=self.gamma,
#            stop_criterion=self.iteration_criterion
#        )
#
#        cdef int n_abstract_actions = np.shape(transition_function)[1]
#
#        # use the bellman equation to solve the q_values
#        cdef np.ndarray q_values = np.zeros(n_abstract_actions)
#        cdef int aa0, sp0
#        for aa0 in range(n_abstract_actions):
#            for sp0 in range(self.n_stim):
#                q_values[aa0] += transition_function[s, aa0, sp0] * (reward_function[sp0] + self.gamma * v[sp0])
#
#        return q_values
#
#    def select_abstract_action_pmf(self, int s, int c, double[:,:,::1] transition_function):
#        cdef np.ndarray[DTYPE_t, ndim=1] q_values = self.get_abstract_action_q_values(s, c, transition_function)
#
#        # we need q-values to properly consider multiple options of equivalent optimality, but we can just always
#        # pass a very high value for the temperature
#        cdef np.ndarray[DTYPE_t, ndim=1] pmf = np.exp(np.array(q_values) * float(self.inverse_temperature))
#        pmf = pmf / np.sum(pmf)
#
#        return pmf
#
#    def get_reward_function(self, int c):
#        cdef int k = self.cluster_assignments[c]
#        cdef RewardCluster cluster = self.clusters[k]
#
#        cdef int n = len(cluster.get_reward_function())
#        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.zeros(n, dtype=DTYPE)
#        cdef int ii
#        for ii in range(n):
#            reward_function[ii] = cluster.get_reward_function()[ii]
#
#        return reward_function
#
#    def get_reward_visits(self, int c):
#        cdef int k = self.cluster_assignments[c]
#        cdef RewardCluster cluster = self.clusters[k]
#        cdef np.ndarray[DTYPE_t, ndim=1] reward_visits = np.asarray(cluster.get_reward_visits())
#
#        return reward_visits
#
#    def get_reward_prediction(self, int c, int sp):
#        cdef int k = self.cluster_assignments[c]
#        cdef RewardCluster cluster = self.clusters[k]
#        cdef double r = cluster.get_reward_prediction(sp)
#        return r
#
#    def set_reward_prior(self, list list_goals):
#        cdef int k
#        cdef RewardCluster cluster
#
#        for k in range(len(self.clusters)):
#            cluster = self.clusters[k]
#            cluster.set_prior(list_goals)
#
#
#cdef class HierarchicalHypothesis(object):
#    cdef int n_stim, n_primitive_actions, n_abstract_actions
#    cdef double gamma, iteration_criterion, inverse_temperature, alpha0, alpha1
#    cdef dict room_assignments, room_clusters, reward_assignments, reward_clusters, mapping_assignments, mapping_clusters
#    cdef double room_log_prior, reward_log_prior, mapping_log_prior, mapping_prior
#
#
#    def __init__(self, int n_stim, int n_primitive_actions, int n_abstract_actions, 
#                 float inverse_temp, float gamma, float stop_criterion, float alpha0, 
#                 float alpha1, float mapping_prior):
#
#        self.n_stim = n_stim
#        self.n_primitive_actions = n_primitive_actions
#        self.n_abstract_actions = n_abstract_actions
#
#        self.inverse_temperature = inverse_temp
#        self.gamma = gamma
#        self.iteration_criterion = stop_criterion
#        self.alpha0 = alpha0
#        self.alpha1 = alpha1
#        
#        self.room_assignments = dict()
#        self.reward_assignments = dict()
#        self.mapping_assignments = dict()
#        
#        self.room_clusters = dict()
#        self.reward_clusters = dict()
#        self.mapping_clusters = dict()
#
#        # initialize posterior
#        self.room_log_prior = 0.
#        self.reward_log_prior = 0.
#        self.mapping_log_prior = 0.
#
#        self.mapping_prior = mapping_prior
#    
#    # reward update
#    def update(self, int c, int sp, int r):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#        cluster.update(c, sp, r)
#        
#    def updating_mapping(self, int c, int a, int aa):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#        cluster.updating_mapping(c, a, aa)
#        
#    def select_abstract_action_pmf(self, int s, int c, double[:,:,::1] transition_function):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#        return cluster.select_abstract_action_pmf(s, c, transition_function)
#
#    def get_reward_function(self, int c):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#        return cluster.get_reward_function(c)
#
#    def get_reward_prediction(self, int c, int sp):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#        return cluster.get_reward_prediction(c,sp)
#
#    def set_reward_prior(self, list list_goals):
#        cdef RewardCluster cluster
#
#        for cluster in self.reward_clusters.values():
#            cluster.set_prior(list_goals)
#
#    def get_mapping_probability(self, int c, int a, int aa):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#        return cluster.get_mapping_probability(c, a, aa)
#
#    cpdef double get_log_posterior(self):
#        cdef double log_posterior = self.room_log_prior + self.mapping_log_prior + self.reward_log_prior
#        log_posterior += sum([cluster.get_reward_log_prior() + cluster.get_mapping_log_prior() + cluster.get_log_likelihood() for cluster in self.room_clusters.values()])
#        return log_posterior
#    
#    def get_room_assignments(self):
#        return self.room_assignments
#    
#    def get_reward_assignments(self, int c):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#
#        cdef list reward_assignments = [self.reward_assignments, cluster.get_reward_assignments()]
#        
#        return reward_assignments
#
#    def get_mapping_assignments(self, int c):
#        cdef int k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster cluster = self.room_clusters[k]
#
#        cdef list mapping_assignments = [self.mapping_assignments, cluster.get_mapping_assignments()]
#        
#        return mapping_assignments
#
#    def add_new_room_context_assignment(self, int c, int k):
#        # check if cluster "k" is already been assigned new cluster
#        if k not in self.room_clusters.keys():
#            # if not, add an new reward cluster
#            self.room_clusters[k] = HierarchicalRoomCluster(self.n_stim, self.n_primitive_actions, 
#                              self.n_abstract_actions, self.inverse_temperature, self.gamma, 
#                              self.iteration_criterion, self.alpha0, self.alpha1, self.mapping_prior)
#
#        self.room_assignments[c] = k  # note, there's no check built in here
#        self.room_log_prior = get_prior_log_probability(self.room_assignments, self.alpha1)
#        
#    def add_new_reward_context_assignment(self, int c, list hierarchical_assignment):
#        cdef int k
#        cdef RewardCluster cluster = None
#        
#        # check if new context has assignment in environment-wide CRP
#        cdef dict env_assignments = hierarchical_assignment[0]
#        if c in env_assignments.keys():
#            k = env_assignments[c]
#            
#            # check if cluster "k" has already been assigned new cluster
#            if k not in self.reward_clusters.keys():
#                # if not, add an new reward cluster
#                self.reward_clusters[k] = RewardCluster(self.n_stim)
#
#            self.reward_assignments[c] = k  # note, there's no check built in here
#            self.reward_log_prior = get_prior_log_probability(self.reward_assignments, self.alpha0)
#            cluster = self.reward_clusters[k]
#            
#        k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
#        assert RoomCluster is not None
#        
#        k = hierarchical_assignment[1][c]
#        RoomCluster.add_new_reward_context_assignment(c, k, cluster)
#        
#    def add_new_mapping_context_assignment(self, int c, list hierarchical_assignment):
#        cdef int k
#        cdef MappingCluster cluster = None
#        
#        # check if new context has assignment in environment-wide CRP
#        cdef dict env_assignments = hierarchical_assignment[0]
#        if c in env_assignments.keys():
#            k = env_assignments[c]
#            
#            # check if cluster "k" has already been assigned new cluster
#            if k not in self.mapping_clusters.keys():
#                # if not, add an new reward cluster
#                self.mapping_clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
#                                                  self.mapping_prior)
#                
#            self.mapping_assignments[c] = k  # note, there's no check built in here
#            self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha0)
#            cluster = self.mapping_clusters[k]
#            
#        k = self.room_assignments[c]
#        cdef HierarchicalRoomCluster RoomCluster = self.room_clusters[k]
#        assert RoomCluster is not None
#        
#        k = hierarchical_assignment[1][c]
#        RoomCluster.add_new_mapping_context_assignment(c, k, cluster)
#
#    def deep_copy(self):
#        cdef HierarchicalHypothesis _h_copy = HierarchicalHypothesis(self.n_stim, 
#                self.n_primitive_actions, self.n_abstract_actions, self.inverse_temperature, 
#                self.gamma, self.iteration_criterion, self.alpha0, self.alpha1, self.mapping_prior)
#
#        _h_copy.room_log_prior = self.room_log_prior
#        _h_copy.reward_log_prior = self.reward_log_prior
#        _h_copy.mapping_log_prior = self.mapping_log_prior
#        
#        _h_copy.room_assignments = dict(self.room_assignments)
#        _h_copy.reward_assignments = dict(self.reward_assignments)
#        _h_copy.mapping_assignments = dict(self.mapping_assignments)
#        
#        _h_copy.reward_clusters = {k:cluster.deep_copy() for k,cluster in self.reward_clusters.iteritems()}
#        _h_copy.mapping_clusters = {k:cluster.deep_copy() for k,cluster in self.mapping_clusters.iteritems()}
#        
#        _h_copy.room_clusters = {k:cluster.deep_copy(self.reward_clusters, _h_copy.reward_clusters, self.mapping_clusters, _h_copy.mapping_clusters) for k,cluster in self.room_clusters.iteritems()}
#
#        return _h_copy
#
#
#cdef class HierarchicalRoomCluster(object):
#    cdef int n_stim, n_primitive_actions, n_abstract_actions
#    cdef double gamma, iteration_criterion, inverse_temperature, alpha0, alpha1
#    cdef dict room_assignments, room_clusters, reward_assignments, reward_clusters, mapping_assignments, mapping_clusters
#    cdef double room_log_prior, reward_log_prior, mapping_log_prior, mapping_prior
#
#    def __init__(self, int n_stim, int n_primitive_actions, int n_abstract_actions, 
#                 float inverse_temp, float gamma, float stop_criterion, float alpha0, 
#                 float alpha1, float mapping_prior):
#        
#        self.n_stim = n_stim
#        self.n_primitive_actions = n_primitive_actions
#        self.n_abstract_actions = n_abstract_actions
#
#        self.inverse_temperature = inverse_temp
#        self.gamma = gamma
#        self.iteration_criterion = stop_criterion
#        self.alpha0 = alpha0
#        self.alpha1 = alpha1
#        
#        self.reward_assignments = dict()
#        self.mapping_assignments = dict()
#        
#        self.reward_clusters = dict()
#        self.mapping_clusters = dict()
#
#        # initialize posterior
#        self.reward_log_prior = 0.
#        self.mapping_log_prior = 0.
#
#        self.mapping_prior = mapping_prior
#        
#        
#    # reward update
#    def update(self, int c, int sp, int r):
#        cdef int k = self.reward_assignments[c]
#        cdef RewardCluster cluster = self.reward_clusters[k]
#        cluster.update(sp, r)
#        
#        
#    def updating_mapping(self, int c, int a, int aa):
#        cdef int k = self.mapping_assignments[c]
#        cdef MappingCluster cluster = self.mapping_clusters[k]
#        cluster.update(a, aa)
#        
#        
#    cpdef np.ndarray[DTYPE_t, ndim=1] get_abstract_action_q_values(self, int s, int c, double[:,:,::1] transition_function):
#        cdef int k = self.reward_assignments[c]
#        cdef RewardCluster cluster = self.reward_clusters[k]
#        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.asarray(cluster.get_reward_function())
#
#        cdef double [:] v
#        v = value_iteration(
#            np.asarray(transition_function),
#            reward_function,
#            gamma=self.gamma,
#            stop_criterion=self.iteration_criterion
#        )
#
#        cdef int n_abstract_actions = np.shape(transition_function)[1]
#
#        # use the bellman equation to solve the q_values
#        cdef np.ndarray q_values = np.zeros(n_abstract_actions)
#        cdef int aa0, sp0
#        for aa0 in range(n_abstract_actions):
#            for sp0 in range(self.n_stim):
#                q_values[aa0] += transition_function[s, aa0, sp0] * (reward_function[sp0] + self.gamma * v[sp0])
#
#        return q_values
#    
#    cpdef np.ndarray[DTYPE_t, ndim=1] select_abstract_action_pmf(self, int s, int c, double[:,:,::1] transition_function):
#        cdef np.ndarray[DTYPE_t, ndim=1] q_values = self.get_abstract_action_q_values(s, c, transition_function)
#
#        # we need q-values to properly consider multiple options of equivalent optimality, but we can just always
#        # pass a very high value for the temperature
#        cdef np.ndarray[DTYPE_t, ndim=1] pmf = np.exp(np.array(q_values) * float(self.inverse_temperature))
#        pmf = pmf / np.sum(pmf)
#
#        return pmf
#
#    cpdef np.ndarray[DTYPE_t, ndim=1] get_reward_function(self, int c):
#        cdef int k = self.reward_assignments[c]
#        cdef RewardCluster cluster = self.reward_clusters[k]
#
#        cdef int n = len(cluster.get_reward_function())
#        cdef np.ndarray[DTYPE_t, ndim=1] reward_function = np.zeros(n, dtype=DTYPE)
#        cdef int ii
#        for ii in range(n):
#            reward_function[ii] = cluster.get_reward_function()[ii]
#
#        return reward_function
#
#    cpdef double get_reward_prediction(self, int c, int sp):
#        cdef int k = self.reward_assignments[c]
#        cdef RewardCluster cluster = self.reward_clusters[k]
#        cdef double r = cluster.get_reward_prediction(sp)
#        return r
#
#    def get_mapping_clusters(self):
#        return self.mapping_clusters
#
#    def get_reward_clusters(self):
#        return self.reward_clusters
#
#    def get_mapping_probability(self, int c, int a, int aa):
#        cdef MappingCluster cluster = self.mapping_clusters[self.mapping_assignments[c]]
#        return cluster.get_mapping_mle(a, aa)
#
#    cpdef double get_log_likelihood(self):
#        cdef double log_likelihood = 0
#        cdef int k, sp, r, aa
#        cdef RewardCluster reward_cluster
#        cdef MappingCluster mapping_cluster
#
#        for reward_cluster in self.reward_clusters.values():
#            log_likelihood += reward_cluster.get_log_likelihood()
#
#        for mapping_cluster in self.mapping_clusters.values():
#            log_likelihood += mapping_cluster.get_log_likelihood()
#
#        return log_likelihood
#    
#    def get_reward_assignments(self):
#        return self.reward_assignments
#    
#    def get_mapping_assignments(self):
#        return self.mapping_assignments
#
#    def add_new_reward_context_assignment(self, int c, int k, RewardCluster cluster):
#        # check if cluster "k" is already been assigned new cluster
#        if k not in self.reward_clusters.keys():
#            # if not, add an new reward cluster
#            assert cluster is not None
#            self.reward_clusters[k] = cluster
#
#        self.reward_assignments[c] = k  # note, there's no check built in here
#        self.reward_log_prior = get_prior_log_probability(self.reward_assignments, self.alpha1)
#
#    def add_new_mapping_context_assignment(self, int c, int k, MappingCluster cluster):
#        # check if cluster "k" is already been assigned new cluster
#        if k not in self.mapping_clusters.keys():
#            # if not, add an new reward cluster
#            assert cluster is not None
#            self.mapping_clusters[k] = cluster
#
#        self.mapping_assignments[c] = k  # note, there's no check built in here
#        self.mapping_log_prior = get_prior_log_probability(self.mapping_assignments, self.alpha1)
#        
#    cpdef double get_reward_log_prior(self):
#        return self.reward_log_prior
#
#    cpdef double get_mapping_log_prior(self):
#        return self.mapping_log_prior
#    
#    def deep_copy(self, dict old_reward_clusters, dict new_reward_clusters, dict old_mapping_clusters, dict new_mapping_clusters):
#        cdef int k, k2
#        cdef RewardCluster reward_cluster, reward_cluster_old
#        cdef MappingCluster mapping_cluster, mapping_cluster_old
#        
#        _cluster_copy = HierarchicalRoomCluster(self.n_stim, self.n_primitive_actions, 
#                                                self.n_abstract_actions, self.inverse_temperature, 
#                                                self.gamma, self.iteration_criterion, self.alpha0, 
#                                                self.alpha1, self.mapping_prior)
#
#        _cluster_copy.reward_log_prior = self.reward_log_prior 
#        _cluster_copy.mapping_log_prior = self.mapping_log_prior
#
#        _cluster_copy.reward_assignments = dict(self.reward_assignments)
#        _cluster_copy.mapping_assignments = dict(self.mapping_assignments)
#        
#        _cluster_copy.reward_clusters = dict()
#        for k, reward_cluster in self.reward_clusters.iteritems():
#            for k2, reward_cluster_old in old_reward_clusters.iteritems():
#                if reward_cluster == reward_cluster_old:
#                    _cluster_copy.reward_clusters[k] = new_reward_clusters[k2]
#                    break
#
#        _cluster_copy.mapping_clusters = dict()
#        for k, mapping_cluster in self.mapping_clusters.iteritems():
#            for k2, mapping_cluster_old in old_mapping_clusters.iteritems():
#                if mapping_cluster == mapping_cluster_old:
#                    _cluster_copy.mapping_clusters[k] = new_mapping_clusters[k2]
#                    break
#
#        return _cluster_copy