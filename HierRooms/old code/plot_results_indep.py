import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# name = "indep_no_structure"
name = "indep_no_sublvl_structure"
# name = "indep3"
# name = "joint"
# name = "ambig"


results = pd.read_pickle("./analyses/HierarchicalRooms_"+name+".pkl")
# mutual_info = pd.read_pickle("./analyses/TaskMutualInfo_"+name+"_hc.pkl")

X0 = results[results['Success'] == True]

# nsims = len(mutual_info)
# sublvl_info = np.array([ mutual_info[ii]['sublvl'][0][-1] for ii in range(nsims) ])
# upper_info = np.array([ mutual_info[ii]['upper room'][0][-1] for ii in range(nsims) ])
# upper_x_sublvl_info = np.array([ mutual_info[ii]['upper mapping x sublvl rewards'][0][-1] for ii in range(nsims) ])

# n_sublvls = len( mutual_info[0]['same sublvl mapping x goal'] )
# same_sublvl_info = np.array( [ [ mutual_info[ii]['same sublvl mapping x goal'][kk][-1] for ii in range(nsims) ] for kk in range(n_sublvls) ] )
# same_sublvl_avg_info = np.mean( same_sublvl_info, axis=0 ) 

# n_doors = len( mutual_info[0]['upper mapping x individual door'] )
# hier_door_info = np.array( [ [ mutual_info[ii]['upper mapping x individual door'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )
# hier_door_avg_info = np.mean( hier_door_info, axis=0 ) 

# flat_subgoal_info = np.array([ mutual_info[ii]['mappings x sublvl goal'][0][-1] for ii in range(nsims) ])
# flat_door_info = np.array( [ [ mutual_info[ii]['mappings x individual door'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )
# flat_door_info_avg = np.mean(flat_door_info, axis=0)

# upper_seq_info = np.array( [ [ mutual_info[ii]['upper sequences'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )



# # get correlations between Indep-Hier steps vs task mutual infos
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

# iterations /= 2

# print 'Upper corr: ', np.corrcoef(upper_info[iterations], indep_hier)[0,1]
# print 'Sublvl corr: ', np.corrcoef(sublvl_info[iterations], indep_hier)[0,1]
# print 'Upper x sublvl corr: ', np.corrcoef(upper_x_sublvl_info[iterations], indep_hier)[0,1]
# print ''

# print 'Upper sequence x door 1: ', np.corrcoef(upper_seq_info[0][iterations], indep_hier)[0,1]
# print 'Upper sequence x door 2: ', np.corrcoef(upper_seq_info[1][iterations], indep_hier)[0,1]
# print 'Upper sequence x door 3: ', np.corrcoef(upper_seq_info[2][iterations], indep_hier)[0,1]
# print 'Upper sequence x door 4: ', np.corrcoef(upper_seq_info[3][iterations], indep_hier)[0,1]



# print ''
# print 'Upper mapping x door 1: ', np.corrcoef(hier_door_info[0][iterations], indep_hier)[0,1]
# print 'Upper mapping x door 2: ', np.corrcoef(hier_door_info[1][iterations], indep_hier)[0,1]
# print 'Upper mapping x door 3: ', np.corrcoef(hier_door_info[2][iterations], indep_hier)[0,1]
# print 'Upper mapping x door 4: ', np.corrcoef(hier_door_info[3][iterations], indep_hier)[0,1]
# print 'Upper mapping x door avg: ', np.corrcoef(hier_door_avg_info[iterations], indep_hier)[0,1]

# print ''
# print 'Flat mappings x door 1: ', np.corrcoef(flat_door_info[0][iterations], indep_hier)[0,1]
# print 'Flat mappings x door 2: ', np.corrcoef(flat_door_info[1][iterations], indep_hier)[0,1]
# print 'Flat mappings x door 3: ', np.corrcoef(flat_door_info[2][iterations], indep_hier)[0,1]
# print 'Flat mappings x door 4: ', np.corrcoef(flat_door_info[3][iterations], indep_hier)[0,1]
# print 'Flat mappings x door avg: ', np.corrcoef(flat_door_info_avg[iterations], indep_hier)[0,1]

# print ''
# print 'Diff info - mappings x door 1: ', np.corrcoef(hier_door_info[0][iterations] - flat_door_info[0][iterations], indep_hier)[0,1]
# print 'Diff info - mappings x door 2: ', np.corrcoef(hier_door_info[1][iterations] - flat_door_info[1][iterations], indep_hier)[0,1]
# print 'Diff info - mappings x door 3: ', np.corrcoef(hier_door_info[2][iterations] - flat_door_info[2][iterations], indep_hier)[0,1]
# print 'Diff info - mappings x door 4: ', np.corrcoef(hier_door_info[3][iterations] - flat_door_info[3][iterations], indep_hier)[0,1]
# print 'Diff info - mappings x door avg: ', np.corrcoef(hier_door_avg_info[iterations] - flat_door_info_avg[iterations], indep_hier)[0,1]

# print ''
# print 'Same sublvl mapping x subgoal 1: ', np.corrcoef(same_sublvl_info[0][iterations], indep_hier)[0,1]
# print 'Same sublvl mapping x subgoal 2: ', np.corrcoef(same_sublvl_info[1][iterations], indep_hier)[0,1]
# print 'Same sublvl mapping x subgoal 3: ', np.corrcoef(same_sublvl_info[2][iterations], indep_hier)[0,1]
# print 'Same sublvl mapping x subgoal avg: ', np.corrcoef(same_sublvl_avg_info[iterations], indep_hier)[0,1]

# print ''
# print 'Flat mapping x flat subgoals: ', np.corrcoef(flat_subgoal_info[iterations], indep_hier)[0,1]

# print ''
# print 'Diff mappings x subgoal 1: ', np.corrcoef(same_sublvl_info[0][iterations] - flat_subgoal_info, indep_hier)[0,1]
# print 'Diff mappings x subgoal 2: ', np.corrcoef(same_sublvl_info[1][iterations] - flat_subgoal_info, indep_hier)[0,1]
# print 'Diff mappings x subgoal 3: ', np.corrcoef(same_sublvl_info[2][iterations] - flat_subgoal_info, indep_hier)[0,1]
# print 'Diff mappings x subgoals: ', np.corrcoef(same_sublvl_avg_info[iterations] - flat_subgoal_info, indep_hier)[0,1]





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

X_max_visits = X_goals.groupby(['Model', 'Room','Iteration'])['Times Seen Context'].max()
X_max_visits = X_max_visits.reset_index()

X_first_by_room = X_first
X_first_by_upper_context = X_upper[X_upper['Times seen door context'] == times_ctx]


# X2 = []
# for ii in range(len(flat_steps)):
#     entry = {'Model': 'Flat', 'Iteration': ii, 'Steps': flat_steps[ii]}
#     X2.append(entry)
    
#     entry = {'Model': 'Indep - Hier', 'Iteration': ii, 'Steps': indep_hier[ii]}
#     X2.append(entry)
# X2 = pd.DataFrame(X2)


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
    # plt.savefig("figs/"+name+'.png', dpi=300, bbox_inches='tight')





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
    # plt.savefig("figs/"+name+'_rooms_traversed.png', dpi=300, bbox_inches='tight')







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
        # plt.savefig("figs/"+name+'_' + model + '_times_seen_context.png', dpi=300, bbox_inches='tight')


    

    # proportion of successful first visits for each context
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
        # plt.savefig("figs/"+name+'_' + model + '_prop_successful_first_visits.png', dpi=300, bbox_inches='tight')






    # proportion of successful first visits (upper)
    X_first_by_upper = X_first_by_room[X_first_by_room['Room'] % 4 == 0]
    n_rooms = max(X_first_by_room['Room'])
    for model in set(X_first_by_upper.Model):
        cc = sns.color_palette('Dark2')
        fig = plt.figure(figsize=(7, 3))

        ax1 = sns.barplot(data=X_first_by_upper[X_first_by_upper['Model']==model], x='Room', y='Reward', order=range(n_rooms))
        ax1.set_ylabel('Proportion of successful first visits')
        ax1.set_xlabel('Room')
        ax1.set_xticklabels(range(0,n_rooms,4))
        ax1.set_title(model)
        ax1.set_ylim([0,1])
    
        plt.tight_layout()
        # plt.savefig("figs/"+name+'_' + model + '_prop_successful_first_visits_upper.png', dpi=300, bbox_inches='tight')



    # proportion of successful first visits (lower)
    X_first_by_sublvl = X_first_by_room[X_first_by_room['Room'] % 4 != 0]
    n_rooms = max(X_first_by_room['Room'])
    for model in set(X_first_by_sublvl.Model):
        cc = sns.color_palette('Dark2')
        fig = plt.figure(figsize=(7, 3))

        ax1 = sns.barplot(data=X_first_by_sublvl[X_first_by_sublvl['Model']==model], x='Room', y='Reward', order=range(n_rooms))
        ax1.set_ylabel('Proportion of successful first visits')
        ax1.set_xlabel('Room')
        ax1.set_xticklabels([ ii for ii in range(n_rooms) if ii % 4 != 0] )
        ax1.set_title(model)
        ax1.set_ylim([0,1])
    
        plt.tight_layout()
        # plt.savefig("figs/"+name+'_' + model + '_prop_successful_first_visits_lower.png', dpi=300, bbox_inches='tight')




    # proportion of successful first visits (upper by sequence)
    n_ctx = max(X_first_by_upper_context['Door context'])
    for model in set(X_first_by_upper_context.Model):
        cc = sns.color_palette('Dark2')
        fig = plt.figure(figsize=(7, 3))

        ax1 = sns.barplot(data=X_first_by_upper_context[X_first_by_upper_context['Model']==model], x='Door context', y='Reward', order=range(n_ctx))
        ax1.set_ylabel('Proportion of successful first visits')
        ax1.set_xlabel('Door context')
        ax1.set_xticklabels(range(n_ctx))
        ax1.set_title(model)
        ax1.set_ylim([0,1])
    
        plt.tight_layout()
        # plt.savefig("figs/"+name+'_' + model + '_prop_successful_first_visits_upper_by_sequence.png', dpi=300, bbox_inches='tight')



    # proportion of successful first visits (upper by first door)
    n_ctx = max(X_first_by_upper_context['Door context'])
    X_first_door = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 0]
    for model in set(X_first_door.Model):
        cc = sns.color_palette('Dark2')
        fig = plt.figure(figsize=(7, 3))

        ax1 = sns.barplot(data=X_first_door[X_first_door['Model']==model], x='Door context', y='Reward', order=range(n_ctx))
        ax1.set_ylabel('Proportion of successful first visits')
        ax1.set_xlabel('Door context')
        ax1.set_xticklabels(range(n_ctx))
        ax1.set_title(model)
        ax1.set_ylim([0,1])
    
        plt.tight_layout()
        # plt.savefig("figs/"+name+'_' + model + '_prop_successful_first_visits_upper_by_first_door.png', dpi=300, bbox_inches='tight')








    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(7, 3)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])


    sns.barplot(data=X_first_by_upper_context, x='Model', y='Reward', ax=ax0, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax0.set_ylabel('Proportion of successful first visits')
    ax0.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    ax0.set_title('Doors')
    ax0.set_ylim([0,1])


    sns.barplot(data=X_first_by_sublvl, x='Model', y='Reward', ax=ax1, 
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    ax1.set_title('Sublvl goals')
    ax1.set_ylim([0,1])
    ax1.set_ylabel('')
    plt.savefig("figs/"+name+ '_prop_successful_first_visits_model_summary.png', dpi=300, bbox_inches='tight')






    # Successful first visit by individual door
    cc = sns.color_palette('Dark2')
    fig = plt.figure(figsize=(13, 13)) 
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.0, 1]) 
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[0,1])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[1,1])


    X_door0 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 0]
    X_door1 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 1]
    X_door2 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 2]
    X_door3 = X_first_by_upper_context[X_first_by_upper_context['Door context'] % 4 == 3]

    sns.barplot(data=X_door0, x='Model', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax0.set_ylabel('Proportion of successful first visits')
    ax0.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    ax0.set_title('First door')
    ax0.set_ylim([0,1])

    sns.barplot(data=X_door1, x='Model', y='Reward', ax=ax1,
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax1.set_ylabel('Proportion of successful first visits')
    ax1.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    ax1.set_title('Second door')
    ax1.set_ylim([0,1])

    sns.barplot(data=X_door2, x='Model', y='Reward', ax=ax2,
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax2.set_ylabel('Proportion of successful first visits')
    ax2.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    ax2.set_title('Third door')
    ax2.set_ylim([0,1])

    sns.barplot(data=X_door3, x='Model', y='Reward', ax=ax3,
                palette='Set2', estimator=np.mean, order=['Flat', 'Independent', 'Hierarchical'])
    ax3.set_ylabel('Proportion of successful first visits')
    ax3.set_xticklabels(['Flat', 'Indep.', 'Hier.'])
    ax3.set_title('Fourth door')
    ax3.set_ylim([0,1])

    plt.savefig("figs/"+name + '_prop_successful_first_visits_by_door.png', dpi=300, bbox_inches='tight')






    # parameters and iteration indexing hard-coded for now
    n_sims = 50
    n_doors = 4
    n_rooms = 16
    diff_prob_success = np.zeros((n_sims,n_doors,n_rooms))
    X_upper_hier = X_first_by_upper_context[X_first_by_upper_context['Model']=='Hierarchical']
    X_upper_ind = X_first_by_upper_context[X_first_by_upper_context['Model']=='Independent']
    col_idx = []
    for kk in range(n_doors):
        for ii in range(n_sims):
            hier_data = X_upper_hier[X_upper_hier['Iteration']==ii]
            ind_data = X_upper_ind[X_upper_ind['Iteration']==ii]
            
            hier_data = hier_data[hier_data['Door context'] % n_doors == kk]
            ind_data = ind_data[ind_data['Door context'] % n_doors == kk]

            if ind_data.shape[0] < n_rooms or hier_data.shape[0] < n_rooms:
                col_idx.append(ii)
                continue
            diff_prob_success[ii][kk] = np.array(hier_data.sort_values(by='Door context')['Reward']) - np.array(ind_data.sort_values(by='Door context')['Reward'])
    col_idx = list(set(col_idx))
    diff_prob_success = np.delete(diff_prob_success,col_idx,axis=0)

    X_diff = pd.DataFrame(data=diff_prob_success.flatten(),columns=['Reward'])
    X_diff['Door'] = 'all'
    
    X_diff0 = pd.DataFrame(data=diff_prob_success[:,0,:].flatten(),columns=['Reward'])
    X_diff0['Door'] = 'door 0'

    X_diff1 = pd.DataFrame(data=diff_prob_success[:,1,:].flatten(),columns=['Reward'])
    X_diff1['Door'] = 'door 1'

    X_diff2 = pd.DataFrame(data=diff_prob_success[:,2,:].flatten(),columns=['Reward'])
    X_diff2['Door'] = 'door 2'

    X_diff3 = pd.DataFrame(data=diff_prob_success[:,3,:].flatten(),columns=['Reward'])
    X_diff3['Door'] = 'door 3'
    
    X_diff = pd.concat([X_diff, X_diff0, X_diff1, X_diff2, X_diff3])
    
    fig = plt.figure(figsize=(7, 5)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
    ax0 = plt.subplot(gs[0,0])
    ax1 = plt.subplot(gs[1,0])

    sns.barplot(data=X_diff, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Diff in success prob')
    ax0.set_ylim([-0.1,0.3])



    diff_prob_success[diff_prob_success < 1] = 0
    X_diff = pd.DataFrame(data=diff_prob_success.flatten(),columns=['Reward'])
    X_diff['Door'] = 'all'
    
    X_diff0 = pd.DataFrame(data=diff_prob_success[:,0,:].flatten(),columns=['Reward'])
    X_diff0['Door'] = 'door 0'

    X_diff1 = pd.DataFrame(data=diff_prob_success[:,1,:].flatten(),columns=['Reward'])
    X_diff1['Door'] = 'door 1'

    X_diff2 = pd.DataFrame(data=diff_prob_success[:,2,:].flatten(),columns=['Reward'])
    X_diff2['Door'] = 'door 2'

    X_diff3 = pd.DataFrame(data=diff_prob_success[:,3,:].flatten(),columns=['Reward'])
    X_diff3['Door'] = 'door 3'
    
    X_diff = pd.concat([X_diff, X_diff0, X_diff1, X_diff2, X_diff3])
    
    sns.barplot(data=X_diff, x='Door', y='Reward', ax=ax1,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax1.set_ylabel('Prob(hier > ind)')
    ax1.set_ylim([0,0.3])
    

    plt.savefig("figs/"+name + '_diff_prob.png', dpi=300, bbox_inches='tight')
    raise
    








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
    # plt.savefig("figs/"+name+'_avg_steps_in_upper.png', dpi=300, bbox_inches='tight')







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
    # plt.savefig("figs/"+name+'_avg_steps_in_sublvl.png', dpi=300, bbox_inches='tight')









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
    # plt.savefig("figs/"+name+'_first_visit.png', dpi=300, bbox_inches='tight')









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
    # plt.savefig("figs/"+name+'_first_visit_upper_' + str(times_ctx) + '.png', dpi=300, bbox_inches='tight')









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
    # plt.savefig("figs/"+name+'_first_visit_sublvl_' + str(times_ctx) +'.png', dpi=300, bbox_inches='tight')







    

    # # histo of flat vs indep-hier
    # cc = sns.color_palette('Dark2')
    # fig = plt.figure(figsize=(7, 3)) 
    # gs = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1]) 
    # ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1])

    # # plot histogram of cumulative steps
    # sns.distplot(X2[X2['Model']=='Indep - Hier']['Steps'], label='Ind-Hier', ax=ax0, color=cc[2])
    # sns.distplot(X2[X2['Model']=='Flat']['Steps'], label='Flat', ax=ax0, color=cc[0])
    # handles, labels = ax0.get_legend_handles_labels()
    # ax0.legend(handles, labels)
    # ax0.set_yticks([])
    # # ax0.set_xlim([0, ax0.get_xlim()[1] ])
    # ax0.set_xlabel('Total Steps')
    
    # sns.despine(offset=2)    
    # ax0.spines['left'].set_visible(False)
    
    
    # # scatterplot of indep-hier vs flat
    # sns.scatterplot(x=flat_steps, y=indep_hier)
    # ax1.set_ylabel('Indep - Hier Agent Steps')
    # ax1.set_xlabel('Flat Agent Steps')
    

    # plt.tight_layout()
    # plt.savefig("figs/"+name+'_Indep-Hier_vs_Flat.png', dpi=300, bbox_inches='tight')
