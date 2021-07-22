#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 03:30:52 2020

@author: rex
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model import plot_results


def plot_one_result(name):
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")

    results_fl = sim_results[sim_results['Model'] == 'Flat']
    results_ic = sim_results[sim_results['Model'] == 'Independent']
    results_jc = sim_results[sim_results['Model'] == 'Joint']
    results_h = sim_results[sim_results['Model'] == 'Hierarchical']
    results_mx = sim_results[sim_results['Model'] == 'Meta']
    
    sim_results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])
    
    # compute cumulative steps to filter out long trials
    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim_results.Model)]
    model = []
    for m in set(sim_results.Model):
        model += [m] * (sim_results[sim_results.Model == m]['Simulation Number'].max() + 1)
    df1 = pd.DataFrame({'Cumulative Steps Taken': np.concatenate(cum_steps),'Model': model})

    # filter out trials exceeding threshold length
    threshold = df1[df1['Model']=='Flat']['Cumulative Steps Taken'].mean() - 3*df1[df1['Model']=='Flat']['Cumulative Steps Taken'].std()
    n_trials = results_fl['Simulation Number'].max()+1
    df1['Failed'] = df1['Cumulative Steps Taken'] > threshold
    df1['Simulation Number'] = np.concatenate([range(n_trials) for m in set(sim_results.Model)])
    
    failed_set = df1[df1['Failed']]

    new_sim_results = results_fl
    for m in ['Independent', 'Joint', 'Hierarchical', 'Meta']:
        truncated_sims = list(set( failed_set[failed_set['Model']==m]['Simulation Number']))

        tmp = sim_results[sim_results['Model']==m]
        tmp_df = tmp[~tmp['Simulation Number'].isin(truncated_sims)]
        new_sim_results = new_sim_results.append(tmp_df, ignore_index=True)

    sim_results = new_sim_results
        
    
    
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx.png', dpi=300, bbox_inches='tight')

    
    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(sim_results.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin.png', dpi=300, bbox_inches='tight')

        
    # separate joint and independent contexts
    # joint_results = sim_results[sim_results['context'] < 5]
    # indep_results = sim_results[(sim_results['context'] > 4) == (sim_results['context'] < 32)]
    # ambig_results = sim_results[(sim_results['context'] > 4) == (sim_results['context'] < 8)]
    # test_joint_results = sim_results[sim_results['context'] == 32]
    # test_indep_results = sim_results[sim_results['context'] > 32]
    # test_indep_results1 = sim_results[sim_results['context'] == 33]
    # test_indep_results2 = sim_results[sim_results['context'] == 34]
    # test_indep_results3 = sim_results[sim_results['context'] == 35]


    # separate joint and independent contexts
    joint_results = sim_results[sim_results['context'] < 4]
    indep_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 85)]
    ambig_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 7)]
    test_joint_results = sim_results[sim_results['context'] == 85]
    test_indep_results = sim_results[sim_results['context'] > 85]
    test_indep_results1 = sim_results[sim_results['context'] == 86]
    test_indep_results2 = sim_results[sim_results['context'] == 87]
    test_indep_results3 = sim_results[sim_results['context'] == 88]
    
    
    # plot_results(joint_results, figsize=(11, 4.5), name='figs/'+name+'_joint.png')

    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=joint_results[joint_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_joint.png', dpi=300, bbox_inches='tight')

    
    df0 = joint_results[joint_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(joint_results.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_joint.png', dpi=300, bbox_inches='tight')




    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=ambig_results[ambig_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_ambig_results.png', dpi=300, bbox_inches='tight')

    
    df0 = ambig_results[ambig_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(ambig_results.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_ambig_results.png', dpi=300, bbox_inches='tight')



        
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=indep_results[indep_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_indep.png', dpi=300, bbox_inches='tight')

    
    df0 = indep_results[indep_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(indep_results.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_indep.png', dpi=300, bbox_inches='tight')



        
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=test_joint_results[test_joint_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_test_joint.png', dpi=300, bbox_inches='tight')

    
    df0 = test_joint_results[test_joint_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_joint_results.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_test_joint.png', dpi=300, bbox_inches='tight')



        
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=test_indep_results[test_indep_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_test_indep.png', dpi=300, bbox_inches='tight')

    
    df0 = test_indep_results[test_indep_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_indep_results.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_test_indep.png', dpi=300, bbox_inches='tight')



        
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=test_indep_results1[test_indep_results1['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_test_indep1.png', dpi=300, bbox_inches='tight')

    
    df0 = test_indep_results1[test_indep_results1['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_indep_results1.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_test_indep1.png', dpi=300, bbox_inches='tight')



        
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=test_indep_results2[test_indep_results2['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_test_indep2.png', dpi=300, bbox_inches='tight')

    
    df0 = test_indep_results2[test_indep_results2['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_indep_results2.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_test_indep2.png', dpi=300, bbox_inches='tight')



        
    with sns.axes_style('ticks'):
        sns.factorplot(x='Times Seen Context', y='n actions taken', data=test_indep_results3[test_indep_results3['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.savefig("figs/"+name+'_ctx_test_indep3.png', dpi=300, bbox_inches='tight')

    
    df0 = test_indep_results3[test_indep_results3['In goal']].groupby(['Model', 'Simulation Number']).sum()
    df1 = pd.DataFrame()
    for m in set(test_indep_results3.Model):
        cum_steps = df0.loc[m]['n actions taken'].values
        model = [m] * len(cum_steps)
        df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
        df1 = df1.append(df2, ignore_index=True)

    sns.set_context('talk')
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax, palette='Set2',
                    order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                    )
        ybar = df1.loc[df1.Model == 'Meta', 'Cumulative Steps Taken'].median()
        ax.plot([-0.5, 4], [ybar, ybar], 'r--')
        ax.set_ylabel('Total Steps')
        ax.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        sns.despine()
        plt.savefig("figs/"+name+'_violin_test_indep3.png', dpi=300, bbox_inches='tight')
        
        


        

name_list = ["AmbigEnvResults_expanded"]
# name_list = ["AmbigEnvResults2"]
# name_list = ["AmbigEnvResults_old"]


for name in name_list:
    print name
    
    plot_one_result(name)

