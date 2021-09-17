import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# name = "indep"
# name = "joint"
# name = "ambig"
name = "HierarchicalRooms_indep_no_structure"


results = pd.read_pickle("./analyses/HierarchicalRooms_"+name+".pkl")
mutual_info = pd.read_pickle("./analyses/TaskMutualInfo_"+name+"_hc.pkl")

X0 = results[results['Success'] == True]

nsims = len(mutual_info)
sublvl_info = np.array([ mutual_info[ii]['sublvl'][0][-1] for ii in range(nsims) ])
upper_info = np.array([ mutual_info[ii]['upper room'][0][-1] for ii in range(nsims) ])
upper_x_sublvl_info = np.array([ mutual_info[ii]['upper mapping x sublvl rewards'][0][-1] for ii in range(nsims) ])

n_sublvls = len( mutual_info[0]['same sublvl mapping x goal'] )
same_sublvl_info = np.array( [ [ mutual_info[ii]['same sublvl mapping x goal'][kk][-1] for ii in range(nsims) ] for kk in range(n_sublvls) ] )
same_sublvl_avg_info = np.mean( same_sublvl_info, axis=0 ) 

n_doors = len( mutual_info[0]['upper mapping x individual door'] )
hier_door_info = np.array( [ [ mutual_info[ii]['upper mapping x individual door'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )
hier_door_avg_info = np.mean( hier_door_info, axis=0 ) 

flat_subgoal_info = np.array([ mutual_info[ii]['mappings x sublvl goal'][0][-1] for ii in range(nsims) ])
flat_door_info = np.array( [ [ mutual_info[ii]['mappings x individual door'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )
flat_door_info_avg = np.mean(flat_door_info, axis=0)


# get correlations between Indep-Hier steps vs task mutual infos
indep_steps = X0[X0['Model']=='Independent']
hier_steps = X0[X0['Model']=='Hierarchical']

iterations = np.array(indep_steps['Iteration'])
hier_steps = hier_steps[hier_steps['Iteration'].isin(iterations)]
iterations = np.array(hier_steps['Iteration'])
indep_steps = indep_steps[indep_steps['Iteration'].isin(iterations)]

indep_steps = indep_steps.sort_values(by='Iteration')
hier_steps = hier_steps.sort_values(by='Iteration')
flat_steps = X0[X0['Model']=='Flat'].sort_values(by='Iteration')

indep_steps = np.array(list(indep_steps['Cumulative Steps']))
hier_steps = np.array(list(hier_steps['Cumulative Steps']))
flat_steps = np.array(list(flat_steps['Cumulative Steps']))

indep_hier = indep_steps-hier_steps
indep_hier = indep_hier

iterations /= 2

print 'Upper corr: ', np.corrcoef(upper_info[iterations], indep_hier)[0,1]
print 'Sublvl corr: ', np.corrcoef(sublvl_info[iterations], indep_hier)[0,1]
print 'Upper x sublvl corr: ', np.corrcoef(upper_x_sublvl_info[iterations], indep_hier)[0,1]

print ''
print 'Upper mapping x door 1: ', np.corrcoef(hier_door_info[0][iterations], indep_hier)[0,1]
print 'Upper mapping x door 2: ', np.corrcoef(hier_door_info[1][iterations], indep_hier)[0,1]
print 'Upper mapping x door 3: ', np.corrcoef(hier_door_info[2][iterations], indep_hier)[0,1]
print 'Upper mapping x door 4: ', np.corrcoef(hier_door_info[3][iterations], indep_hier)[0,1]
print 'Upper mapping x door avg: ', np.corrcoef(hier_door_avg_info[iterations], indep_hier)[0,1]

print ''
print 'Flat mappings x door 1: ', np.corrcoef(flat_door_info[0][iterations], indep_hier)[0,1]
print 'Flat mappings x door 2: ', np.corrcoef(flat_door_info[1][iterations], indep_hier)[0,1]
print 'Flat mappings x door 3: ', np.corrcoef(flat_door_info[2][iterations], indep_hier)[0,1]
print 'Flat mappings x door 4: ', np.corrcoef(flat_door_info[3][iterations], indep_hier)[0,1]
print 'Flat mappings x door avg: ', np.corrcoef(flat_door_info_avg[iterations], indep_hier)[0,1]

print ''
print 'Diff info - mappings x door 1: ', np.corrcoef((hier_door_info[0] - flat_door_info[0])[iterations], indep_hier)[0,1]
print 'Diff info - mappings x door 2: ', np.corrcoef((hier_door_info[1] - flat_door_info[1])[iterations], indep_hier)[0,1]
print 'Diff info - mappings x door 3: ', np.corrcoef((hier_door_info[2] - flat_door_info[2])[iterations], indep_hier)[0,1]
print 'Diff info - mappings x door 4: ', np.corrcoef((hier_door_info[3] - flat_door_info[3])[iterations], indep_hier)[0,1]
print 'Diff info - mappings x door avg: ', np.corrcoef((hier_door_avg_info - flat_door_info_avg)[iterations], indep_hier)[0,1]

print ''
print 'Same sublvl mapping x subgoal 1: ', np.corrcoef(same_sublvl_info[0][iterations], indep_hier)[0,1]
print 'Same sublvl mapping x subgoal 2: ', np.corrcoef(same_sublvl_info[1][iterations], indep_hier)[0,1]
print 'Same sublvl mapping x subgoal 3: ', np.corrcoef(same_sublvl_info[2][iterations], indep_hier)[0,1]
print 'Same sublvl mapping x subgoal avg: ', np.corrcoef(same_sublvl_avg_info[iterations], indep_hier)[0,1]

print ''
print 'Flat mapping x flat subgoals: ', np.corrcoef(flat_subgoal_info[iterations], indep_hier)[0,1]

print ''
print 'Diff mappings x subgoal 1: ', np.corrcoef((same_sublvl_info[0] - flat_subgoal_info)[iterations], indep_hier)[0,1]
print 'Diff mappings x subgoal 2: ', np.corrcoef((same_sublvl_info[1] - flat_subgoal_info)[iterations], indep_hier)[0,1]
print 'Diff mappings x subgoal 3: ', np.corrcoef((same_sublvl_info[2] - flat_subgoal_info)[iterations], indep_hier)[0,1]
print 'Diff mappings x subgoals: ', np.corrcoef((same_sublvl_avg_info - flat_subgoal_info)[iterations], indep_hier)[0,1]





sns.set_context('paper', font_scale=1.25)
from matplotlib import gridspec

# 
X_goals = results[results['In Goal']]
X_upper = X_goals[X_goals['Room'] % 4 == 0]
X_lower = X_goals[X_goals['Room'] % 4 != 0]

times_ctx = 1
X_first = X_goals[X_goals['Times Seen Context'] == times_ctx]
X_first_upper = X_first[X_first['Room'] % 4 == 0]
X_first_lower = X_first[X_first['Room'] % 4 != 0]

X_max_visits = X_goals.groupby(['Model', 'Room'])['Times Seen Context'].max()
X_max_visits = X_max_visits.reset_index()

X_first_by_room = X_first.groupby(['Model','Room'])['Reward'].sum()/nsims
X_first_by_room = X_first_by_room.reset_index()


X2 = []
for ii in range(len(flat_steps)):
    entry = {'Model': 'Flat', 'Iteration': ii, 'Steps': flat_steps[ii]}
    X2.append(entry)
    
for ii in range(len(indep_hier)):
    entry = {'Model': 'Indep - Hier', 'Iteration': ii, 'Steps': indep_hier[ii]}
    X2.append(entry)
X2 = pd.DataFrame(X2)


with sns.axes_style('ticks'):
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X0[X0['Model']=='Hierarchical']['Cumulative Steps'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X0[X0['Model']=='Independent']['Cumulative Steps'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X0[X0['Model']=='Flat']['Cumulative Steps'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlabel('Total Steps')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)




    # plot bar chart of cumulative steps
    sns.barplot(data=X0, x='Model', y='Cumulative Steps', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Total Steps')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    
    plt.tight_layout()
    plt.savefig("figs/"+name+'.png', dpi=300, bbox_inches='tight')





    # final trial numbers
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of final trial numbers
    sns.distplot(X0[X0['Model']=='Hierarchical']['Trial Number'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X0[X0['Model']=='Independent']['Trial Number'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X0[X0['Model']=='Flat']['Trial Number'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlabel('Number of rooms traversed')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)


    # plot bar chart of final trial numbers
    sns.barplot(data=X0, x='Model', y='Trial Number', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Number of rooms traversed')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    
    plt.tight_layout()
    plt.savefig("figs/"+name+'_rooms_traversed.png', dpi=300, bbox_inches='tight')







    # max visits for each context
    n_rooms = max(X_max_visits['Room'])
    for model in set(X_max_visits.Model):
        cc = sns.color_palette('Dark2')
        fig = plt.figure(figsize=(7, 3))

        ax1 = sns.barplot(data=X_max_visits[X_max_visits['Model']==model], x='Room', y='Times Seen Context', order=range(n_rooms))
        ax1.set_ylabel('Times Seen Context')
        ax1.set_xlabel('Room')
        ax1.set_xticklabels(range(n_rooms))
        ax1.set_title(model)
    
        plt.tight_layout()
        plt.savefig("figs/"+name+'_' + model + '_times_seen_context.png', dpi=300, bbox_inches='tight')


    

    # proportion of successful first visits
    n_rooms = max(X_first_by_room['Room'])
    for model in set(X_first_by_room.Model):
        cc = sns.color_palette('Dark2')
        fig = plt.figure(figsize=(7, 3))

        ax1 = sns.barplot(data=X_first_by_room[X_first_by_room['Model']==model], x='Room', y='Reward', order=range(n_rooms))
        ax1.set_ylabel('Proportion of successful first visits')
        ax1.set_xlabel('Room')
        ax1.set_xticklabels(range(n_rooms))
        ax1.set_title(model)
        ax1.set_ylim([0,1])
    
        plt.tight_layout()
        plt.savefig("figs/"+name+'_' + model + '_prop_successful_first_visits.png', dpi=300, bbox_inches='tight')






    # steps in upper vs sublvls
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X_upper[X_upper['Model']=='Hierarchical']['Steps Taken'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X_upper[X_upper['Model']=='Independent']['Steps Taken'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X_upper[X_upper['Model']=='Flat']['Steps Taken'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlim([0, 20 ])
    ax0.set_xlabel('Steps in upper')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)


    # plot bar chart of cumulative steps
    h_sims_hc = len(X_upper[X_upper['Model']=='Hierarchical'])
    h_sims_ic = len(X_upper[X_upper['Model']=='Independent'])
    h_sims_fl = len(X_upper[X_upper['Model']=='Flat'])
    
    X1 = pd.DataFrame({
        'Steps Taken': np.concatenate([
                X_upper[X_upper['Model']=='Hierarchical']['Steps Taken'].values,
                X_upper[X_upper['Model']=='Independent']['Steps Taken'].values,
                X_upper[X_upper['Model']=='Flat']['Steps Taken'].values, 
            ]),
        'Model': ['Hierarchical'] * h_sims_hc + ['Independent'] * h_sims_ic + ['Flat'] * h_sims_fl,
    })
    sns.barplot(data=X_upper, x='Model', y='Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Steps Taken')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

    plt.tight_layout()
    plt.savefig("figs/"+name+'_avg_steps_in_upper.png', dpi=300, bbox_inches='tight')







    # steps in upper vs sublvls
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X_lower[X_lower['Model']=='Hierarchical']['Steps Taken'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X_lower[X_lower['Model']=='Independent']['Steps Taken'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X_lower[X_lower['Model']=='Flat']['Steps Taken'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, 30 ])
    ax0.set_xlabel('Steps in sublvl')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)

    
    # plot bar chart of cumulative steps
    h_sims_hc = len(X_lower[X_lower['Model']=='Hierarchical'])
    h_sims_ic = len(X_lower[X_lower['Model']=='Independent'])
    h_sims_fl = len(X_lower[X_lower['Model']=='Flat'])
    
    X1 = pd.DataFrame({
        'Steps Taken': np.concatenate([
                X_lower[X_lower['Model']=='Hierarchical']['Steps Taken'].values,
                X_lower[X_lower['Model']=='Independent']['Steps Taken'].values,
                X_lower[X_lower['Model']=='Flat']['Steps Taken'].values, 
            ]),
        'Model': ['Hierarchical'] * h_sims_hc + ['Independent'] * h_sims_ic + ['Flat'] * h_sims_fl,
    })
    sns.barplot(data=X1, x='Model', y='Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Steps Taken')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

    plt.tight_layout()
    plt.savefig("figs/"+name+'_avg_steps_in_sublvl.png', dpi=300, bbox_inches='tight')









    # steps in first visit of context
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X_first[X_first['Model']=='Hierarchical']['Steps Taken'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X_first[X_first['Model']=='Independent']['Steps Taken'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X_first[X_first['Model']=='Flat']['Steps Taken'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, 30 ])
    ax0.set_xlabel('Steps in first visit of a context')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)

    
    # plot bar chart of cumulative steps
    h_sims_hc = len(X_first[X_first['Model']=='Hierarchical'])
    h_sims_ic = len(X_first[X_first['Model']=='Independent'])
    h_sims_fl = len(X_first[X_first['Model']=='Flat'])
    
    X1 = pd.DataFrame({
        'Steps Taken': np.concatenate([
                X_first[X_first['Model']=='Hierarchical']['Steps Taken'].values,
                X_first[X_first['Model']=='Independent']['Steps Taken'].values,
                X_first[X_first['Model']=='Flat']['Steps Taken'].values, 
            ]),
        'Model': ['Hierarchical'] * h_sims_hc + ['Independent'] * h_sims_ic + ['Flat'] * h_sims_fl,
    })
    sns.barplot(data=X1, x='Model', y='Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Steps Taken')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

    plt.tight_layout()
    plt.savefig("figs/"+name+'_first_visit.png', dpi=300, bbox_inches='tight')









    # steps in first visit of context -- upper
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X_first_upper[X_first_upper['Model']=='Hierarchical']['Steps Taken'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X_first_upper[X_first_upper['Model']=='Independent']['Steps Taken'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X_first_upper[X_first_upper['Model']=='Flat']['Steps Taken'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, 30 ])
    ax0.set_xlabel('Steps in first visit of upper')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)

    
    # plot bar chart of cumulative steps
    h_sims_hc = len(X_first_upper[X_first_upper['Model']=='Hierarchical'])
    h_sims_ic = len(X_first_upper[X_first_upper['Model']=='Independent'])
    h_sims_fl = len(X_first_upper[X_first_upper['Model']=='Flat'])
    
    X1 = pd.DataFrame({
        'Steps Taken': np.concatenate([
                X_first_upper[X_first_upper['Model']=='Hierarchical']['Steps Taken'].values,
                X_first_upper[X_first_upper['Model']=='Independent']['Steps Taken'].values,
                X_first_upper[X_first_upper['Model']=='Flat']['Steps Taken'].values, 
            ]),
        'Model': ['Hierarchical'] * h_sims_hc + ['Independent'] * h_sims_ic + ['Flat'] * h_sims_fl,
    })
    sns.barplot(data=X1, x='Model', y='Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Steps Taken')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

    plt.tight_layout()
    plt.savefig("figs/"+name+'_first_visit_upper_' + str(times_ctx) + '.png', dpi=300, bbox_inches='tight')









    # steps in first visit of context -- sublvl
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X_first_lower[X_first_lower['Model']=='Hierarchical']['Steps Taken'], label='Hier.', ax=ax0, color=cc[2])
    sns.distplot(X_first_lower[X_first_lower['Model']=='Independent']['Steps Taken'], label='Ind.', ax=ax0, color=cc[1])
    sns.distplot(X_first_lower[X_first_lower['Model']=='Flat']['Steps Taken'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    ax0.set_xlim([0, 30 ])
    ax0.set_xlabel('Steps in first visit of sublvl')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)

    
    # plot bar chart of cumulative steps
    h_sims_hc = len(X_first_lower[X_first_lower['Model']=='Hierarchical'])
    h_sims_ic = len(X_first_lower[X_first_lower['Model']=='Independent'])
    h_sims_fl = len(X_first_lower[X_first_lower['Model']=='Flat'])
    
    X1 = pd.DataFrame({
        'Steps Taken': np.concatenate([
                X_first_lower[X_first_lower['Model']=='Hierarchical']['Steps Taken'].values,
                X_first_lower[X_first_lower['Model']=='Independent']['Steps Taken'].values,
                X_first_lower[X_first_lower['Model']=='Flat']['Steps Taken'].values, 
            ]),
        'Model': ['Hierarchical'] * h_sims_hc + ['Independent'] * h_sims_ic + ['Flat'] * h_sims_fl,
    })
    sns.barplot(data=X1, x='Model', y='Steps Taken', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Steps Taken')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])

    plt.tight_layout()
    plt.savefig("figs/"+name+'_first_visit_sublvl_' + str(times_ctx) +'.png', dpi=300, bbox_inches='tight')







    

    # histo of flat vs indep-hier
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # plot histogram of cumulative steps
    sns.distplot(X2[X2['Model']=='Indep - Hier']['Steps'], label='Ind-Hier', ax=ax0, color=cc[2])
    sns.distplot(X2[X2['Model']=='Flat']['Steps'], label='Flat', ax=ax0, color=cc[0])
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels)
    ax0.set_yticks([])
    # ax0.set_xlim([0, ax0.get_xlim()[1] ])
    ax0.set_xlabel('Total Steps')
    
    sns.despine(offset=2)    
    ax0.spines['left'].set_visible(False)
    
    
    # scatterplot of indep-hier vs flat
    sns.scatterplot(x=flat_steps, y=indep_hier)
    ax1.set_ylabel('Indep - Hier Agent Steps')
    ax1.set_xlabel('Flat Agent Steps')
    

    plt.tight_layout()
    plt.savefig("figs/"+name+'_Indep-Hier_vs_Flat.png', dpi=300, bbox_inches='tight')
