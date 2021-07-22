#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:45:13 2020

@author: rex
"""

from copy import deepcopy
from cpython cimport bool

def augment_assignments(list cluster_assignments, int new_context, bool flags=False):
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


def augment_sublvl_assignments(list cluster_assignments, int n_sublvls, int new_context, bool flags=False):
    # context: upper rooms always have context k*(n_sublvls+1),
    # first sublvl always has context k*(n_sublvls+1)+1
    # second sublvl always has context k*(n_sublvls+1)+2, etc
    
    cdef list _cluster_assignments = list()
    cdef list new_cluster_flags = list() # tracks which new assignments involved creating new cluster
    
    cdef list sublvl_clustering, _clustering_copy, _used_idx, _unused_idx
    cdef int n_sublvl_clustering, ii, k, kk
    cdef dict subassignment, _unused_subassignments, _new_sub
    cdef list new_subassignments, new_flags, _new_assignment, _unused_without_ii

    cdef int sublvl = new_context % (n_sublvls+1)
    cdef int room_ctx = new_context - sublvl
    
    assert sublvl > 0
    
    for sublvl_clustering in cluster_assignments:
        n_sublvl_clustering = len(sublvl_clustering)
        assert n_sublvl_clustering <= n_sublvls
        _clustering_copy = deepcopy(sublvl_clustering)
            
        if n_sublvl_clustering < n_sublvls:
            _clustering_copy = _clustering_copy+[dict()]
            n_sublvl_clustering += 1
            
        # note which sublvl cluster assignments include a sublvl from
        # current room and which do not
        _used_idx = list()
        _unused_idx = list()
        for ii, subassignment in enumerate(_clustering_copy):
            if any( k+room_ctx in subassignment.keys() for k in range(1,n_sublvl_clustering+1) ):
                _used_idx += [ii]
            else:
                _unused_idx += [ii]

            
        for ii, k in enumerate(_unused_idx):
            _unused_subassignments = _clustering_copy[k]
                
            # for each unused sublvl_clustering, generate an augmented sublvl_clustering
            new_subassignments, new_flags = augment_assignments([_unused_subassignments], new_context, True)
            new_cluster_flags += new_flags
                
            # for each new sublvl_clustering, collate with other sublvl_clusterings to
            # create an augmented cluster assignment
            for _new_sub in new_subassignments:
                _new_assignment = [_new_sub]
                    
                for kk in _used_idx:
                # add copies of 'used' subassignments to the new assignment
                    _new_assignment.append(_clustering_copy[kk].copy())
                        
                _unused_without_ii = _unused_idx[:ii]+_unused_idx[ii+1:]
                for kk in _unused_without_ii:
                # add copies of 'unused' subassignments to the new assignment
                    _new_assignment.append(_clustering_copy[kk].copy())
                        
                # sort _new_assignment according to first n_sublvl contexts
                _new_assignment.sort(key = (lambda x: min(x.keys())))

                _cluster_assignments.append(_new_assignment)

    assert len(_cluster_assignments) == len(new_cluster_flags)

    if flags:
        return _cluster_assignments, new_cluster_flags
    else:
        return _cluster_assignments


def augment_hierarchical_assignments(list cluster_assignments, int n_sublvls, int new_context):
    # cluster_assignments is the hierarchy of cluster assignments stored as a dict
    # cluster_assignments[0] is root layer, cluster_assignments[1] is 2nd layer, etc
    # lower layer contains sublvl cluster assignments
    #
    # context: upper rooms always have context k*(n_sublvls+1),
    # first sublvl always has context k*(n_sublvls+1)+1
    # second sublvl always has context k*(n_sublvls+1)+2, etc
    
    cdef list assignment, _new_hier, upper_layers, lowest_layer
    cdef list _sublvl_cluster_assignments, new_cluster_flags
    cdef list sub_assignment, _new_upper_hier_list, 
    cdef int ii
    
    cdef int sublvl = new_context % (n_sublvls+1)    
    assert sublvl > 0
    
    cdef int n_layers = len(cluster_assignments[0])
    
    cdef list _cluster_assignments = list()

    for assignment in cluster_assignments:
        if len(assignment[0]) == 0:
            # empty Chinese restaurant franchise
            _new_hier = [{new_context: 0} for ii in range(n_layers-1)]
            _new_hier.append([{new_context: 0}])
            _cluster_assignments = [_new_hier]
        else:
            upper_layers = assignment[:-1]
            lowest_layer = deepcopy(assignment[-1])
        
            _sublvl_cluster_assignments, new_cluster_flags = augment_sublvl_assignments(
                    [lowest_layer], n_sublvls, new_context, True)
        
            for ii, sub_assignment in enumerate(_sublvl_cluster_assignments):
                if new_cluster_flags[ii]:
                    # lowest level of hierarchy created new cluster, so expand upper levels
                    _new_upper_hier_list = expand_CRF(upper_layers, new_context)
                    for _new_hier in _new_upper_hier_list:
                        _new_hier.append(deepcopy(sub_assignment))
                        _cluster_assignments.append(_new_hier)
                else:
                    _new_hier = deepcopy(upper_layers)
                    _new_hier.append(sub_assignment)
                    _cluster_assignments.append(_new_hier)
                
    return _cluster_assignments

# recursively expands the Chinese restaurant franchise (CRF) across layers
def expand_CRF(list old_layers, int new_context):
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
                    _new_hier.append(deepcopy(assignment))
                    _cluster_assignments.append(_new_hier)
            else:
                _new_hier = deepcopy(old_layers[:-1])
                _new_hier.append(assignment)
                _cluster_assignments.append(_new_hier)

    return _cluster_assignments
