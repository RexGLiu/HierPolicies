#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:09:43 2020

@author: rex
"""

import numpy as np

class UpperDoorCluster(object):

    def __init__(self, n_goals, goal_prior):
        self.n_goals = n_goals
        self.goal_prior = goal_prior

        # rewards!
        self.set_visits =  n_goals * goal_prior
        self.goal_rewards_received = np.ones((n_goals,n_goals)) * goal_prior
        self.goal_reward_probability = np.ones((n_goals,n_goals)) * (1.0 / n_goals)

    def update(self, seq, goal, r):
        self.set_visits += 1.0
        
        baseline = 1.0 / (self.n_goals+1)

        if r > 0:     
            if seq + 1 < self.n_goals:
                r0 = 1.0 / (self.n_goals - seq - 1)

            self.goal_reward_probability[seq+1:self.n_goals,goal] = 0
            self.goal_reward_probability[seq,:] = 0
            self.goal_reward_probability[seq,goal] = 1
            
            probability_subset = self.goal_reward_probability[seq+1:self.n_goals,:]
            idx = np.logical_and(1.0 > probability_subset, probability_subset > baseline)
            probability_subset[idx] = r0
            
        else:
            assert self.goal_reward_probability[seq,goal] < 1.0
            self.goal_reward_probability[seq,goal] = 0
            
            count = 0
            
            # remaining unvisited goals for current seq has equal probability of being true goal
            for g0 in range(self.n_goals):
                if self.goal_reward_probability[seq,g0] > baseline:
                    # count non-zero entries of given sequence.
                    # Anything below baseline should be regarded as zero.
                    count += 1
                
            # by elimination, one remaining goal must be true goal
            # set that goal to have prob 1 and renormalise probs for remaining seqs
            if count == 1:
                if seq + 1 < self.n_goals:
                    r0 = 1.0 / (self.n_goals - seq - 1)
                    
                g1 = np.argmax(self.goal_reward_probability[seq,:])

                self.goal_reward_probability[seq+1:self.n_goals,g1] = 0
                self.goal_reward_probability[seq,:] = 0
                self.goal_reward_probability[seq,g1] = 1
            
                for s0 in range(seq+1,self.n_goals):
                    for g0 in range(self.n_goals):
                        if 1.0 > self.goal_reward_probability[s0,g0] and self.goal_reward_probability[s0,g0] > baseline:
                            self.goal_reward_probability[s0,g0] = r0
            else: 
                r0 = 1.0 / count
            
                for g0 in range(self.n_goals):
                    if 1.0 > self.goal_reward_probability[seq,g0] and self.goal_reward_probability[seq,g0] > baseline:
                        # condition overwrites only probabilities that are 1 or 0 
                        self.goal_reward_probability[seq,g0] = r0
                    
                     
