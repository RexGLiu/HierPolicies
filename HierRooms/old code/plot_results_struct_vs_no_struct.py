import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

results = pd.read_pickle("./analyses/HierarchicalRooms_indep_no_structure.pkl")
results2 = pd.read_pickle("./analyses/HierarchicalRooms_indep_no_sublvl_structure.pkl")

X_goals = results[results['In Goal']]
X_upper = X_goals[X_goals['Room'] % 4 == 0]
X_first_by_upper_context = X_upper[X_upper['Times seen door context'] == 1]
X_upper_hier = X_first_by_upper_context[X_first_by_upper_context['Model']=='Hierarchical']
X_upper_ind = X_first_by_upper_context[X_first_by_upper_context['Model']=='Independent']

X_goals2 = results2[results2['In Goal']]
X_upper2 = X_goals2[X_goals2['Room'] % 4 == 0]
X_first_by_upper_context2 = X_upper2[X_upper2['Times seen door context'] == 1]
X_upper_hier2 = X_first_by_upper_context2[X_first_by_upper_context2['Model']=='Hierarchical']
X_upper_ind2 = X_first_by_upper_context2[X_first_by_upper_context2['Model']=='Independent']


n_sims = 50
n_doors = 4
n_rooms = 16



diff_prob_success = np.zeros((n_sims,n_doors,n_rooms))
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
prob_hier = np.array(diff_prob_success)
prob_hier[prob_hier < 1] = 0
prob_ind = -np.array(diff_prob_success)
prob_ind[prob_ind < 1] = 0


diff_prob_success2 = np.zeros((n_sims,n_doors,n_rooms))
col_idx = []
for kk in range(n_doors):
    for ii in range(n_sims):
        hier_data2 = X_upper_hier2[X_upper_hier2['Iteration']==ii]
        ind_data2 = X_upper_ind2[X_upper_ind2['Iteration']==ii]
            
        hier_data2 = hier_data2[hier_data2['Door context'] % n_doors == kk]
        ind_data2 = ind_data2[ind_data2['Door context'] % n_doors == kk]

        if ind_data2.shape[0] < n_rooms or hier_data2.shape[0] < n_rooms:
            col_idx.append(ii)
            continue
        diff_prob_success2[ii][kk] = np.array(hier_data2.sort_values(by='Door context')['Reward']) - np.array(ind_data2.sort_values(by='Door context')['Reward'])

col_idx = list(set(col_idx))
diff_prob_success2 = np.delete(diff_prob_success2,col_idx,axis=0)
prob_hier2 = np.array(diff_prob_success2)
prob_hier2[prob_hier2 < 1] = 0
prob_ind2 = -np.array(diff_prob_success2)
prob_ind2[prob_ind2 < 1] = 0

diff_prob_success_diff = np.mean(np.mean(diff_prob_success2,axis=2),axis=0) - np.mean(np.mean(diff_prob_success,axis=2),axis=0)
prob_hier_diff = np.mean(np.mean(prob_hier2,axis=2),axis=0) - np.mean(np.mean(prob_hier,axis=2),axis=0)
prob_ind_diff = np.mean(np.mean(prob_ind2,axis=2),axis=0) - np.mean(np.mean(prob_ind,axis=2),axis=0)



tmp = pd.DataFrame(data=diff_prob_success.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=diff_prob_success[:,0,:].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=diff_prob_success[:,1,:].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=diff_prob_success[:,2,:].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=diff_prob_success[:,3,:].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Diff_Prob = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])


tmp = pd.DataFrame(data=diff_prob_success2.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=diff_prob_success2[:,0,:].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=diff_prob_success2[:,1,:].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=diff_prob_success2[:,2,:].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=diff_prob_success2[:,3,:].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Diff_Prob2 = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])


tmp = pd.DataFrame(data=prob_hier.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=prob_hier[:,0,:].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=prob_hier[:,1,:].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=prob_hier[:,2,:].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=prob_hier[:,3,:].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Prob_Hier = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])


tmp = pd.DataFrame(data=prob_hier2.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=prob_hier2[:,0,:].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=prob_hier2[:,1,:].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=prob_hier2[:,2,:].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=prob_hier2[:,3,:].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Prob_Hier2 = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])


tmp = pd.DataFrame(data=prob_ind.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=prob_ind[:,0,:].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=prob_ind[:,1,:].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=prob_ind[:,2,:].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=prob_ind[:,3,:].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Prob_Ind = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])


tmp = pd.DataFrame(data=prob_ind2.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=prob_ind2[:,0,:].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=prob_ind2[:,1,:].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=prob_ind2[:,2,:].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=prob_ind2[:,3,:].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Prob_Ind2 = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])



tmp = pd.DataFrame(data=diff_prob_success_diff.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=diff_prob_success_diff[0].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=diff_prob_success_diff[1].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=diff_prob_success_diff[2].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=diff_prob_success_diff[3].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Diff_Prob_Diff = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])


tmp = pd.DataFrame(data=prob_hier_diff.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=prob_hier_diff[0].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=prob_hier_diff[1].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=prob_hier_diff[2].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=prob_hier_diff[3].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Prob_Hier_Diff = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])


tmp = pd.DataFrame(data=prob_ind_diff.flatten(),columns=['Reward'])
tmp['Door'] = 'all'
    
tmp0 = pd.DataFrame(data=prob_ind_diff[0].flatten(),columns=['Reward'])
tmp0['Door'] = 'door 0'

tmp1 = pd.DataFrame(data=prob_ind_diff[1].flatten(),columns=['Reward'])
tmp1['Door'] = 'door 1'

tmp2 = pd.DataFrame(data=prob_ind_diff[2].flatten(),columns=['Reward'])
tmp2['Door'] = 'door 2'

tmp3 = pd.DataFrame(data=prob_ind_diff[3].flatten(),columns=['Reward'])
tmp3['Door'] = 'door 3'
    
Prob_Ind_Diff = pd.concat([tmp, tmp0, tmp1, tmp2, tmp3])







sns.set_context('paper', font_scale=1.25)

with sns.axes_style('ticks'):
    fig = plt.figure(figsize=(21, 21)) 
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1], width_ratios=[1,1,1])

    ax0 = plt.subplot(gs[0,0])
    sns.barplot(data=Diff_Prob, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Diff in success prob')
    ax0.set_ylim([-0.1,0.3])
    ax0.set_title('No structure')

    ax0 = plt.subplot(gs[0,1])
    sns.barplot(data=Diff_Prob2, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Diff in success prob')
    ax0.set_ylim([-0.1,0.3])
    ax0.set_title('Door structure')

    ax0 = plt.subplot(gs[0,2])
    sns.barplot(data=Diff_Prob_Diff, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Diff in success prob')
    ax0.set_ylim([-0.1,0.1])
    ax0.set_title('Door structure - No structure')

    ax0 = plt.subplot(gs[1,0])
    sns.barplot(data=Prob_Hier, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Prob(hier > ind)')
    ax0.set_ylim([-0.1,0.3])

    ax0 = plt.subplot(gs[1,1])
    sns.barplot(data=Prob_Hier2, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Prob(hier > ind)')
    ax0.set_ylim([-0.1,0.3])

    ax0 = plt.subplot(gs[1,2])
    sns.barplot(data=Prob_Hier_Diff, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Prob(hier > ind)')
    ax0.set_ylim([-0.1,0.1])


    ax0 = plt.subplot(gs[2,0])
    sns.barplot(data=Prob_Ind, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Prob(ind > hier)')
    ax0.set_ylim([-0.1,0.3])

    ax0 = plt.subplot(gs[2,1])
    sns.barplot(data=Prob_Ind2, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Prob(ind > hier)')
    ax0.set_ylim([-0.1,0.3])

    ax0 = plt.subplot(gs[2,2])
    sns.barplot(data=Prob_Ind_Diff, x='Door', y='Reward', ax=ax0,
                palette='Set2', estimator=np.mean, order=['all', 'door 0', 'door 1', 'door 2', 'door 3'])
    ax0.set_ylabel('Prob(ind > hier)')
    ax0.set_ylim([-0.1,0.1])



    plt.savefig("figs/Struct_vs_NoStruct_Probs.png", dpi=300, bbox_inches='tight')

