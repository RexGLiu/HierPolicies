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


def plot_one_result(name, file_idx):
    sim_results = pd.read_pickle("./analyses/"+name+".pkl")

    results_fl = sim_results[sim_results['Model'] == 'Flat']
    results_ic = sim_results[sim_results['Model'] == 'Independent']
    results_jc = sim_results[sim_results['Model'] == 'Joint']
    results_h = sim_results[sim_results['Model'] == 'Hierarchical']
    results_mx = sim_results[sim_results['Model'] == 'Meta']
    
    sim_results = pd.concat([results_fl,results_ic,results_jc,results_h,results_mx])
    
    df0 = sim_results[sim_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    cum_steps = [df0.loc[m]['n actions taken'].values for m in set(sim_results.Model)]
    model = []
    for m in set(sim_results.Model):
        model += [m] * (sim_results[sim_results.Model == m]['Simulation Number'].max() + 1)
    df1 = pd.DataFrame({'Cumulative Steps Taken': np.concatenate(cum_steps),'Model': model})


    # filter out trials exceeding threshold length
    threshold = df1[df1['Model']=='Flat']['Cumulative Steps Taken'].mean() - 2*df1[df1['Model']=='Flat']['Cumulative Steps Taken'].std()
    n_trials = results_fl['Simulation Number'].max()+1
    df1['Failed'] = df1['Cumulative Steps Taken'] > threshold
    df1['Simulation Number'] = np.concatenate([range(n_trials) for m in set(sim_results.Model)])
    
    for m in set(sim_results.Model):
        if m == 'Flat':
            continue
        
        truncated_sims = set( df1[df1['Failed']][df1['Model']==m]['Simulation Number'])
        tmp = sim_results[sim_results['Model']==m]
        sim_results[sim_results['Model']==m] = tmp[~tmp.isin(truncated_sims)]
        
    df1 = df1[(df1.Model=='Flat') | (~df1.Failed)]

    plt.figure(figsize=fig_size)
    with sns.axes_style('ticks'):
        sns.pointplot(x='Times Seen Context', y='n actions taken', data=sim_results[sim_results['In goal']], 
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        plt.gca().get_legend().remove()
        plt.gca().set_yticks(range(0,50,10))
        plt.gca().set_ylim(0,45)
        # plt.legend(prop={'size': 14})
        # if file_idx==0:
        #     plt.legend(prop={'size': 14})
        # else:
        #     plt.gca().get_legend().remove()
        sns.despine()
        plt.tight_layout()
    #     plt.savefig("figs/"+name+"_ctx.png", dpi=300, bbox_inches='tight')


    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])
    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+"_violin.png", dpi=300, bbox_inches='tight')


    if file_idx == 0:
        # separate joint and independent contexts
        joint_results = sim_results[sim_results['context'] < 4]
        indep_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 16)]
        ambig_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 6)]
        train_results = sim_results[sim_results['context'] < 16]
        test_joint_results = sim_results[sim_results['context'] == 16]
        test_indep_results = sim_results[sim_results['context'] > 16]
        test_indep_results1 = sim_results[sim_results['context'] == 17]
        test_indep_results2 = sim_results[sim_results['context'] == 18]

    else:
        # separate joint and independent contexts
        joint_results = sim_results[sim_results['context'] < 4]
        indep_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 34)]
        ambig_results = sim_results[(sim_results['context'] > 3) == (sim_results['context'] < 6)]
        train_results = sim_results[sim_results['context'] < 34]
        test_joint_results = sim_results[sim_results['context'] == 34]
        test_indep_results = sim_results[sim_results['context'] > 34]
        test_indep_results1 = sim_results[sim_results['context'] == 35]
        test_indep_results2 = sim_results[sim_results['context'] == 36]


    # plt.figure(figsize=fig_size)
    # with sns.axes_style('ticks'):
    #     sns.pointplot(x='Times Seen Context', y='n actions taken', data=joint_results[joint_results['In goal']],
    #                     units='Simulation Number', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     # plt.legend(prop={'size': 14})
    #     plt.gca().get_legend().remove()
    #     plt.gca().set_yticks(range(0,50,10))
    #     plt.gca().set_ylim(0,45)
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_ctx_joint.png', dpi=300, bbox_inches='tight')

    
    # df0 = joint_results[joint_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(joint_results.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_joint.png', dpi=300, bbox_inches='tight')




    # plt.figure(figsize=fig_size)
    # with sns.axes_style('ticks'):
    #     sns.pointplot(x='Times Seen Context', y='n actions taken', data=ambig_results[ambig_results['In goal']],
    #                     units='Simulation Number', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     plt.gca().get_legend().remove()
    #     # plt.legend(prop={'size': 14})
    #     plt.gca().set_yticks(range(0,50,10))
    #     plt.gca().set_ylim(0,45)
        
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_ctx_ambig_results.png', dpi=300, bbox_inches='tight')

    
    # df0 = ambig_results[ambig_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(ambig_results.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_ambig_results.png', dpi=300, bbox_inches='tight')



        
    # plt.figure(figsize=fig_size)
    # with sns.axes_style('ticks'):
    #     sns.pointplot(x='Times Seen Context', y='n actions taken', data=indep_results[indep_results['In goal']],
    #                     units='Simulation Number', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     plt.gca().get_legend().remove()
    #     plt.legend(prop={'size': 14})
    #     plt.gca().set_yticks(range(0,50,10))
    #     plt.gca().set_ylim(0,45)
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_ctx_indep.png', dpi=300, bbox_inches='tight')

    
    # df0 = indep_results[indep_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(indep_results.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_indep.png', dpi=300, bbox_inches='tight')



        
    plt.figure(figsize=fig_size)
    with sns.axes_style('ticks'):
        sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_joint_results[test_joint_results['In goal']],
                        units='Simulation Number', hue='Model', estimator=np.mean,
                        palette='Set2')
        sns.despine()
        plt.gca().get_legend().remove()
        plt.legend(prop={'size': 14})
        plt.gca().set_yticks(range(0,50,10))
        plt.gca().set_ylim(0,45)
        plt.tight_layout()
        plt.savefig("figs/"+name+'_ctx_test_joint.png', dpi=300, bbox_inches='tight')

    
    # df0 = test_joint_results[test_joint_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(test_joint_results.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_test_joint.png', dpi=300, bbox_inches='tight')



        
    # plt.figure(figsize=fig_size)
    # with sns.axes_style('ticks'):
    #     sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_indep_results[test_indep_results['In goal']],
    #                     units='Simulation Number', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     plt.gca().get_legend().remove()
    #     # plt.legend(prop={'size': 14})
    #     plt.gca().set_yticks(range(0,50,10))
    #     plt.gca().set_ylim(0,45)
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_ctx_test_indep.png', dpi=300, bbox_inches='tight')

    
    # df0 = test_indep_results[test_indep_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(test_indep_results.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_test_indep.png', dpi=300, bbox_inches='tight')



        
    # plt.figure(figsize=fig_size)
    # with sns.axes_style('ticks'):
    #     sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_indep_results1[test_indep_results1['In goal']],
    #                     units='Simulation Number', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     plt.gca().get_legend().remove()
    #     # plt.legend(prop={'size': 14})
    #     plt.gca().set_yticks(range(0,50,10))
    #     plt.gca().set_ylim(0,45)
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_ctx_test_indep1.png', dpi=300, bbox_inches='tight')

    
    # df0 = test_indep_results1[test_indep_results1['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(test_indep_results1.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_test_indep1.png', dpi=300, bbox_inches='tight')



        
    # plt.figure(figsize=fig_size)
    # with sns.axes_style('ticks'):
    #     sns.pointplot(x='Times Seen Context', y='n actions taken', data=test_indep_results2[test_indep_results2['In goal']],
    #                     units='Simulation Number', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     plt.gca().get_legend().remove()
    #     # plt.legend(prop={'size': 14})
    #     plt.gca().set_yticks(range(0,50,10))
    #     plt.gca().set_ylim(0,45)
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_ctx_test_indep2.png', dpi=300, bbox_inches='tight')

    
    # df0 = test_indep_results2[test_indep_results2['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(test_indep_results2.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_test_indep2.png', dpi=300, bbox_inches='tight')



        
    # plt.figure(figsize=fig_size)
    # with sns.axes_style('ticks'):
    #     sns.pointplot(x='Times Seen Context', y='n actions taken', data=train_results[train_results['In goal']],
    #                     units='Simulation Number', hue='Model', estimator=np.mean,
    #                     palette='Set2')
    #     sns.despine()
    #     # plt.gca().get_legend().remove()
    #     plt.legend(prop={'size': 14})
    #     plt.gca().set_yticks(range(0,50,10))
    #     plt.gca().set_ylim(0,45)
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_ctx_train.png', dpi=300, bbox_inches='tight')

    
    # df0 = train_results[train_results['In goal']].groupby(['Model', 'Simulation Number']).sum()
    # df1 = pd.DataFrame()
    # for m in set(train_results.Model):
    #     cum_steps = df0.loc[m]['n actions taken'].values
    #     model = [m] * len(cum_steps)
    #     df2 = pd.DataFrame({'Cumulative Steps Taken': cum_steps,'Model': model})
    #     df1 = df1.append(df2, ignore_index=True)

    # plt.figure(figsize=fig_size)
    # sns.set_context('talk')
    # with sns.axes_style('ticks'):
    #     sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', palette='Set2',
    #                 order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
    #                 )
    #     ybar = df1.loc[df1.Model == 'Hierarchical', 'Cumulative Steps Taken'].median()
    #     plt.plot([-0.5, 4.5], [ybar, ybar], 'r--')
    #     plt.gca().set_ylabel('Total Steps')
    #     plt.gca().set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

    #     sns.despine()
    #     plt.tight_layout()
    #     plt.savefig("figs/"+name+'_violin_train.png', dpi=300, bbox_inches='tight')






name_list = ["AmbigEnvResults_truncated_1", "AmbigEnvResults_truncated_3"]
# name_list = ["AmbigEnvResults_truncated_3"]
n_files = len(name_list)


fig_size = (5, 4.5)

for idx, name in enumerate(name_list):
    print name
    
    plot_one_result(name, idx)
