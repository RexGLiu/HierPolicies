#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:04:10 2020

@author: rex
"""

import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

from joblib import Parallel, delayed
import multiprocessing

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# custom libraries used
from models.grid_world import Experiment
from models.agents import IndependentClusterAgent, JointClusteringAgent, FlatAgent, MetaAgent, HierarchicalAgent
from models.experiment_designs.experiment4 import gen_task_param
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

np.random.seed(0)

# pre generate a set of tasks for consistency. 
list_tasks = [gen_task_param() for _ in range(n_sims)]

# pre draw the alphas for consistency
list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale)) 
              for _ in range(n_sims)]

list_beta = [np.exp(scipy.random.normal(loc=beta_mu, scale=beta_scale))
                for _ in range(n_sims)]

num_cores = multiprocessing.cpu_count()

def sim_agent(AgentClass, name='None', flat=False, meta=False, hier=False):
    print name
    
    tasks = tqdm(enumerate(list_tasks), total=len(list_tasks))
    
    results = Parallel(n_jobs=num_cores)(delayed(simulate_one)(AgentClass, ii, task_args, task_kwargs, name, flat, meta, hier)
                                       for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)))

    return pd.concat(results)


def simulate_one(AgentClass, ii, task_args, task_kwargs, name='None', flat=False, meta=False, hier=False):
    if not flat:
        if hier:
            agent_kwargs = dict(alpha0=2, alpha1=1, sample_tau = 1, inv_temp=inv_temp, mapping_prior=mapping_prior,
                            goal_prior=goal_prior)
        else:
            agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=list_beta[ii], mapping_prior=mapping_prior,
                                goal_prior=goal_prior)
    else:
        agent_kwargs = dict(inv_temp=list_beta[ii], goal_prior=goal_prior, mapping_prior=mapping_prior)
            
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

results_ic = sim_agent(IndependentClusterAgent, name='Independent')
results_jc = sim_agent(JointClusteringAgent, name='Joint')
results_h = sim_agent(HierarchicalAgent, name='Hierarchical', hier=True)
results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
results = pd.concat([results_ic, results_jc, results_h, results_fl, results_meta])

results.to_pickle("./AmbigEnv5Results.pkl")

