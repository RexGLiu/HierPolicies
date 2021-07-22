#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:45:13 2020

@author: rex
"""

from cpython cimport bool

cpdef list enumerate_assignments(int max_context_number):
    """
     enumerate all possible assignments of contexts to clusters for a fixed number of contexts. Has the
     hard assumption that the first context belongs to cluster #1, to remove redundant assignments that
     differ in labeling.

    :param max_context_number: int
    :return: list of lists, each a function that takes in a context id number and returns a cluster id number
    """
    cdef list cluster_assignments = [{}]  # context 0 is always in cluster 1
    cdef int contextNumber

    for contextNumber in range(0, max_context_number):
        cluster_assignments = augment_assignments(cluster_assignments, contextNumber)

    return cluster_assignments


cpdef augment_assignments(list cluster_assignments, int new_context, bool flags=False):
    cdef list _cluster_assignments = list()
    cdef list new_cluster_flags = list()
    cdef dict assignment, _assignment_copy
    cdef list new_list
    cdef int k
    
    if (len(cluster_assignments) == 0) | (len(cluster_assignments[0]) == 0):
        _cluster_assignments.append({new_context: 0})
        new_cluster_flags.append(True)
    else:
        for assignment in cluster_assignments:
            new_list = list()
            for k in range(0, max(assignment.values()) + 2):
                _assignment_copy = assignment.copy()
                _assignment_copy[new_context] = k
                new_list.append(_assignment_copy)
            _cluster_assignments += new_list
            new_cluster_flags += [False]*(max(assignment.values()) + 1)
            new_cluster_flags += [True]

    assert len(_cluster_assignments) == len(new_cluster_flags)
    
    if flags:
        return _cluster_assignments, new_cluster_flags
    else:
        return _cluster_assignments


cpdef list augment_hierarchical_assignments(list cluster_assignments, int new_context):
    # cluster_assignments is the hierarchy of cluster assignments stored as a dict
    # cluster_assignments[0] is root layer, cluster_assignments[1] is 2nd layer, etc
    
    cdef list assignment, _new_hier, _new_assigment
    cdef int ii
    
    cdef int n_layers = len(cluster_assignments[0])
    
    cdef list _cluster_assignments = list()

    for assignment in cluster_assignments:
        if len(assignment[0]) == 0:
            # empty Chinese restaurant franchise
            _new_hier = [{new_context: 0} for ii in range(n_layers)]
            _cluster_assignments = [_new_hier]
        else:
            _new_assigment = expand_CRF(assignment, new_context)
            _cluster_assignments += _new_assigment
            
    return _cluster_assignments

# recursively expands the Chinese restaurant franchise (CRF) across layers
cdef list expand_CRF(list old_layers, int new_context):
    cdef list new_layer_assignments, new_cluster_flags, _new_upper_hier_list
    cdef dict assignment
    cdef int ii

    new_layer_assignments, new_cluster_flags = augment_assignments([old_layers[-1]], new_context, True)
    
    cdef list _cluster_assignments = list()
    cdef int n_layers = len(old_layers)

    if n_layers == 1:
        _cluster_assignments = [ [assignment] for assignment in new_layer_assignments]
    else:
        for ii, assignment in enumerate(new_layer_assignments):
            if new_cluster_flags[ii]:
                # lowest level of hierarchy created new cluster, so expand upper levels
                _new_upper_hier_list = expand_CRF(old_layers[:-1], new_context)
                for _new_hier in _new_upper_hier_list:
                    _assignment_copy = {k:c for k,c in assignment.iteritems()}
                    _new_hier.append(_assignment_copy)
                    _cluster_assignments.append(_new_hier)
            else:
                _new_hier = [ {k:c for k,c in _layer.iteritems()} for _layer in old_layers[:-1]]
                _new_hier.append(assignment)
                _cluster_assignments.append(_new_hier)

    return _cluster_assignments
