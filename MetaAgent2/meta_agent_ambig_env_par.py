#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:04:10 2020

@author: rex
"""

from mpi4py import MPI

from datetime import datetime
import random

import scipy
import pandas as pd
import numpy as np

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# custom libraries used
from models.grid_world import Experiment
from models.agents import IndependentClusterAgent, JointClusteringAgent, FlatAgent, MetaAgent, HierarchicalAgent
from models.experiment_designs.experiment1 import gen_task_param
sns.set_context('paper', font_scale=1.5)


n_sims = 2500

# alpha is sample from the distribution
# log(alpha) ~ N(alpha_mu, alpha_scale)
alpha_mu = -0.5
alpha_scale = 1.0

beta_mu = 2.0
beta_scale = 0.5

inv_temp = 10.0
goal_prior = 1e-10 
mapping_prior = 1e-10
pruning_threshold = 500.0
evaluate = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.Get_size()

# random.seed(datetime.now())
random.seed(500)
seed0 = random.randint(0,10000)*(rank+1)
seed = int(seed0)
np.random.seed(seed)


def simulate_one(AgentClass, ii, task_args, task_kwargs, alpha, beta, name='None', flat=False, meta=False, hier=False):
    if not flat:
        if hier:
            agent_kwargs = dict(alpha0=1., alpha1=1, sample_tau = 1, inv_temp=inv_temp, mapping_prior=mapping_prior,
                            goal_prior=goal_prior)
        else:
            agent_kwargs = dict(alpha=alpha, inv_temp=beta, mapping_prior=mapping_prior,
                                goal_prior=goal_prior)
    else:
        agent_kwargs = dict(inv_temp=beta, goal_prior=goal_prior, mapping_prior=mapping_prior)
            
    if meta:
        p = np.random.uniform(0, 1)
        agent_kwargs['mix_biases'] = [np.log(p), np.log(1-p)]
        agent_kwargs['update_new_c_only'] = True

    agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)
    
    _res = None
    while _res is None:
        _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)
    _res[u'Model'] = [name] * len(_res)
    _res[u'Iteration'] = [ii] * len(_res)

    return _res



# generate list of tasks for each sim and scatter data
# master process
if rank == 0:
    # pre generate a set of tasks for consistency. 
    list_tasks = [gen_task_param() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale)) 
                  for _ in range(n_sims)]

    list_beta = [np.exp(scipy.random.normal(loc=beta_mu, scale=beta_scale))
                 for _ in range(n_sims)]

    q, r = divmod(n_sims, n_procs)
    counts = [q + 1 if p < r else q for p in range(n_procs)]    

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(n_procs)]
    ends = [sum(counts[:p+1]) for p in range(n_procs)]

    list_tasks = [list_tasks[starts[p]:ends[p]] for p in range(n_procs)]
    list_alpha = [list_alpha[starts[p]:ends[p]] for p in range(n_procs)]
    list_beta = [list_beta[starts[p]:ends[p]] for p in range(n_procs)]

# worker process
else:
    list_tasks = None
    list_alpha = None
    list_beta = None

list_tasks = comm.scatter(list_tasks, root=0)
list_alpha = comm.scatter(list_alpha, root=0)
list_beta = comm.scatter(list_beta, root=0)

n_tasks = len(list_tasks)

q, r = divmod(n_sims, n_procs)
if rank < r:
    iteration_offset = (q+1)*rank
else:
    iteration_offset = (q+1)*r + q*(rank-r)

# Hierarchical Agent
results_hc = []
for kk in range(n_tasks):
    task_args, task_kwargs = list_tasks[kk]
    
    _result = simulate_one(HierarchicalAgent, iteration_offset+kk, task_args, task_kwargs, alpha=1., beta=inv_temp, name='Hierarchical', hier=True)
    results_hc.append(_result)

# Independent Agent
results_ic = []
for kk in range(n_tasks):
    task_args, task_kwargs = list_tasks[kk]
    alpha = list_alpha[kk]
    beta = list_beta[kk]
    
    _result = simulate_one(IndependentClusterAgent, iteration_offset+kk, task_args, task_kwargs, alpha, beta, name='Independent')
    results_ic.append(_result)

# Joint Agent
results_jc = []
for kk in range(n_tasks):
    task_args, task_kwargs = list_tasks[kk]
    alpha = list_alpha[kk]
    beta = list_beta[kk]
    
    _result = simulate_one(JointClusteringAgent, iteration_offset+kk, task_args, task_kwargs, alpha, beta, name='Joint')
    results_jc.append(_result)
    
# Flat Agent
results_fl = []
for kk in range(n_tasks):
    task_args, task_kwargs = list_tasks[kk]
    beta = list_beta[kk]
    
    _result = simulate_one(FlatAgent, iteration_offset+kk, task_args, task_kwargs, alpha=None, beta=beta, name='Flat', flat=True)
    results_fl.append(_result)
    
# Meta Agent
results_m = []
for kk in range(n_tasks):
    task_args, task_kwargs = list_tasks[kk]
    alpha = list_alpha[kk]
    beta = list_beta[kk]
    
    _result = simulate_one(MetaAgent, iteration_offset+kk, task_args, task_kwargs, alpha, beta, name='Meta', meta=True)
    results_m.append(_result)


_results_hc = comm.gather(results_hc, root=0)
_results_ic = comm.gather(results_ic, root=0)
_results_jc = comm.gather(results_jc, root=0)
_results_fl = comm.gather(results_fl, root=0)
_results_m = comm.gather(results_m, root=0)

if rank == 0:
    results_hc = []
    for result in _results_hc:
        results_hc += result
    results_hc = pd.concat(results_hc)

    results_ic = []
    for result in _results_ic:
        results_ic += result
    results_ic = pd.concat(results_ic)

    results_jc = []
    for result in _results_jc:
        results_jc += result
    results_jc = pd.concat(results_jc)

    results_fl = []
    for result in _results_fl:
        results_fl += result
    results_fl = pd.concat(results_fl)

    results_m = []
    for result in _results_m:
        results_m += result
    results_m = pd.concat(results_m)

    results = pd.concat([results_ic, results_jc, results_hc, results_fl, results_m])
    
    results.to_pickle("./Gen2GenEnvResults.pkl")

