import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from model.comp_rooms import sequential_mutual_info
from scipy.stats import ttest_ind as ttest
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2



intercept_baseline = -np.log(13./3)

def intercept_to_baseline(intercept):
    K = 1./(1 + np.exp(-intercept))
    disc = np.sqrt(1-4*K)
    return ( (1 + disc)/2 , (1 - disc)/2 )


def likelihood_ratio_test(X, y, model):
    full_log_likelihood = 2*log_loss(y, model.predict_proba(X), normalize=False)
    
    p1 = np.sum(y)/y.size
    intercept_model = p1*np.ones((y.size,2))
    intercept_model[:,0] = 1-p1
    intercept_log_likelihood = 2*log_loss(y, intercept_model, normalize=False)
    
    const_model = np.ones((y.size,2))
    const_model[:,1] = 0
    const_log_likelihood = 2*log_loss(y, const_model, normalize=False)
    
    intercept_const_log_ratio = const_log_likelihood - intercept_log_likelihood
    full_intercept_log_ratio = intercept_log_likelihood - full_log_likelihood
    
    N_dof = model.coef_.size
    p_intercept = 1-chi2.cdf(full_intercept_log_ratio, df=N_dof)
    p_const = 1-chi2.cdf(intercept_const_log_ratio, df=1)
    
    return full_intercept_log_ratio, intercept_const_log_ratio, p_intercept, p_const


def BIC(X,Y,model):
    k = model.coef_.size + model.intercept_.size
    n = Y.size
    BIC_full = k*np.log(n) + 2*log_loss(Y, model.predict_proba(X), normalize=False)

    k = model.intercept_.size
    p1 = np.sum(Y)/Y.size
    intercept_model = p1*np.ones((Y.size,2))
    intercept_model[:,0] = 1-p1
    BIC_intercept = k*np.log(n) + 2*log_loss(Y, intercept_model, normalize=False)

    const_model = np.ones((Y.size,2))
    const_model[:,1] = 0
    BIC_const = 2*log_loss(Y, const_model, normalize=False)

    return BIC_full, BIC_intercept, BIC_const


# residual structure removed
results = pd.read_pickle("./analyses/HierarchicalRooms_indep_no_structure.pkl")
mutual_info = pd.read_pickle("./analyses/TaskMutualInfo_indep_no_structure.pkl")

# residual structure present
results2 = pd.read_pickle("./analyses/HierarchicalRooms_indep_no_sublvl_structure.pkl")
mutual_info2 = pd.read_pickle("./analyses/TaskMutualInfo_indep_no_sublvl_structure.pkl")




X_goals = results[results['In Goal']]
X_upper = X_goals[X_goals['Room'] % 4 == 0]
X_first_by_upper_context = X_upper[X_upper['Times seen door context'] == 1]
X_upper_hier = X_first_by_upper_context[X_first_by_upper_context['Model']=='Hierarchical']
X_upper_ind = X_first_by_upper_context[X_first_by_upper_context['Model']=='Independent']



n_sims = 50
n_doors = 4
n_rooms = 16
diff_prob_success = np.zeros((n_sims,n_doors,n_rooms))
hier_prob_success = np.zeros((n_sims,n_doors,n_rooms))
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
        hier_prob_success[ii][kk] = np.array(hier_data.sort_values(by='Door context')['Reward'])

col_idx = list(set(col_idx))
diff_prob_success = np.delete(diff_prob_success,col_idx,axis=0)
hier_prob_success = np.delete(hier_prob_success,col_idx,axis=0)
        
    


upper_seq_frac_info = np.array( [ [ [ mutual_info[ii]['upper sequences normalised cumulative info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors) ] for ii in range(1,n_sims) ] )
upper_seq_info = np.array( [ [ [ mutual_info[ii]['upper sequences cumulative info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors) ] for ii in range(1,n_sims) ] )
upper_seq_cond_frac_info = np.array( [ [ [ mutual_info[ii]['upper sequences normalised conditional cum info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors-1) ] for ii in range(1,n_sims) ] )
upper_seq_cond_info = np.array( [ [ [ mutual_info[ii]['upper sequences conditional cum info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors-1) ] for ii in range(1,n_sims) ] )










for ii in range(n_doors):
    print 'Frac cumulative info x success prob', ii, ":", np.corrcoef(diff_prob_success[:,ii,:].flatten(), upper_seq_frac_info[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success.flatten(), upper_seq_frac_info.flatten())[0,1]

for ii in range(n_doors):
    print 'Cumulative info x success prob', ii, ":", np.corrcoef(diff_prob_success[:,ii,:].flatten(), upper_seq_info[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success.flatten(), upper_seq_info.flatten())[0,1]

for ii in range(n_doors-1):
    print 'Frac conditional cum info x success prob', ii+1, ":", np.corrcoef(diff_prob_success[:,ii+1,:].flatten(), upper_seq_cond_frac_info[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success[:,1:,:].flatten(), upper_seq_cond_frac_info.flatten())[0,1]

for ii in range(n_doors-1):
    print 'Conditional cum info x success prob', ii+1, ":", np.corrcoef(diff_prob_success[:,ii+1,:].flatten(), upper_seq_cond_info[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success[:,1:,:].flatten(), upper_seq_cond_info.flatten())[0,1]
print ''



for ii in range(n_doors):
    print 'Frac cumulative info x success prob', ii, ":", np.corrcoef(hier_prob_success[:,ii,:].flatten(), upper_seq_frac_info[:,ii,:].flatten())[0,1]
print np.corrcoef(hier_prob_success.flatten(), upper_seq_frac_info.flatten())[0,1]

for ii in range(n_doors):
    print 'Cumulative info x success prob', ii, ":", np.corrcoef(hier_prob_success[:,ii,:].flatten(), upper_seq_info[:,ii,:].flatten())[0,1]
print np.corrcoef(hier_prob_success.flatten(), upper_seq_info.flatten())[0,1]

for ii in range(n_doors-1):
    print 'Frac conditional cum info x success prob', ii+1, ":", np.corrcoef(hier_prob_success[:,ii+1,:].flatten(), upper_seq_cond_frac_info[:,ii,:].flatten())[0,1]
print np.corrcoef(hier_prob_success[:,1:,:].flatten(), upper_seq_cond_frac_info.flatten())[0,1]

for ii in range(n_doors-1):
    print 'Conditional cum info x success prob', ii+1, ":", np.corrcoef(hier_prob_success[:,ii+1,:].flatten(), upper_seq_cond_info[:,ii,:].flatten())[0,1]
print np.corrcoef(hier_prob_success[:,1:,:].flatten(), upper_seq_cond_info.flatten())[0,1]
print ''


# t-test and logistic regression
equal_var = False

Y = np.array(diff_prob_success.flatten())
Y[Y < 1] = 0
Y = Y.reshape((Y.size,1))

print np.sum(Y)/Y.size, 1-np.sum(Y)/Y.size 



for ii in range(n_doors):
    diff_prob_success_subset = diff_prob_success[:,ii,:]
    info_subset = upper_seq_frac_info[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Fractional cumulative info, door " + str(ii) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_frac_info[diff_prob_success > 0]
frac_info_miss = upper_seq_frac_info[diff_prob_success < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Fractional cumulative info, all doors:", t_val, p_val

X = upper_seq_frac_info.flatten()
X = zscore(X)
X = X.reshape((X.size,1))
clf = LogisticRegression(C=1E16).fit(X,Y)
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X,Y)
Y_pred = clf.predict(X)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y,clf)
print 'BIC: ', BIC(X,Y,clf)
print ''

for ii in range(n_doors):
    diff_prob_success_subset = diff_prob_success[:,ii,:]
    info_subset = upper_seq_info[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Absolute cumulative info, door " + str(ii) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_info[diff_prob_success > 0]
frac_info_miss = upper_seq_info[diff_prob_success < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Absolute cumulative info, all doors:", t_val, p_val

X = upper_seq_info.flatten()
X = X.reshape((X.size,1))
X = zscore(X)
clf = LogisticRegression(C=1E16).fit(X,Y)
dev = likelihood_ratio_test(X,Y,clf)
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X,Y)
Y_pred = clf.predict(X)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y,clf)
print 'BIC: ', BIC(X,Y,clf)
print ''


for ii in range(n_doors-1):
    diff_prob_success_subset = diff_prob_success[:,ii+1,:]
    info_subset = upper_seq_cond_frac_info[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Absolute conditional cumulative info, door " + str(ii+1) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_cond_frac_info[diff_prob_success[:,1:,:] > 0]
frac_info_miss = upper_seq_cond_frac_info[diff_prob_success[:,1:,:] < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Absolute conditional cumulative info, all doors:", t_val, p_val
print ''


for ii in range(n_doors-1):
    diff_prob_success_subset = diff_prob_success[:,ii+1,:]
    info_subset = upper_seq_cond_info[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Fractional conditional cumulative info, door " + str(ii+1) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_cond_info[diff_prob_success[:,1:,:] > 0]
frac_info_miss = upper_seq_cond_info[diff_prob_success[:,1:,:] < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Fractional conditional cumulative info, all doors:", t_val, p_val
print ''
















X_goals2 = results2[results2['In Goal']]
X_upper2 = X_goals2[X_goals2['Room'] % 4 == 0]
X_first_by_upper_context2 = X_upper2[X_upper2['Times seen door context'] == 1]
X_upper_hier2 = X_first_by_upper_context2[X_first_by_upper_context2['Model']=='Hierarchical']
X_upper_ind2 = X_first_by_upper_context2[X_first_by_upper_context2['Model']=='Independent']



# parameters and iteration indexing hard-coded for now
diff_prob_success2 = np.zeros((n_sims,n_doors,n_rooms))
hier_prob_success2 = np.zeros((n_sims,n_doors,n_rooms))
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
        hier_prob_success2[ii][kk] = np.array(hier_data2.sort_values(by='Door context')['Reward'])

col_idx = list(set(col_idx))
diff_prob_success2 = np.delete(diff_prob_success2,col_idx,axis=0)
hier_prob_success2 = np.delete(hier_prob_success2,col_idx,axis=0)





upper_seq_frac_info2 = np.array( [ [ [ mutual_info2[ii]['upper sequences normalised cumulative info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors) ] for ii in range(n_sims) ] )
upper_seq_info2 = np.array( [ [ [ mutual_info2[ii]['upper sequences cumulative info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors) ] for ii in range(n_sims) ] )
upper_seq_cond_frac_info2 = np.array( [ [ [ mutual_info2[ii]['upper sequences normalised conditional cum info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors-1) ] for ii in range(n_sims) ] )
upper_seq_cond_info2 = np.array( [ [ [ mutual_info2[ii]['upper sequences conditional cum info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors-1) ] for ii in range(n_sims) ] )

upper_seq_frac_info_combined = np.concatenate((upper_seq_frac_info,upper_seq_frac_info2),axis=0)
upper_seq_info_combined = np.concatenate((upper_seq_info,upper_seq_info2),axis=0)
upper_seq_cond_frac_info_combined = np.concatenate((upper_seq_cond_frac_info,upper_seq_cond_frac_info2),axis=0)
upper_seq_cond_info_combined = np.concatenate((upper_seq_cond_info,upper_seq_cond_info2),axis=0)

diff_prob_success_combined = np.concatenate((diff_prob_success,diff_prob_success2),axis=0)
hier_prob_success_combined = np.concatenate((hier_prob_success,hier_prob_success2),axis=0)



print "Unstructured prob diff: ", np.mean(np.mean(diff_prob_success,2),0)
print "Structured prob diff: ", np.mean(np.mean(diff_prob_success2,2),0)
print "Difference: ", np.mean(np.mean(diff_prob_success2,2),0) - np.mean(np.mean(diff_prob_success,2),0)
print ''
tmp = diff_prob_success > 0
tmp2 = diff_prob_success2 > 0
print "Unstructured P(hier > ind): ", np.mean(np.mean(tmp,2),0)
print "Structured P(hier > ind): ", np.mean(np.mean(tmp2,2),0)
print "Difference: ", np.mean(np.mean(tmp2,2),0) - np.mean(np.mean(tmp,2),0)
print ''
tmp = diff_prob_success < 0
tmp2 = diff_prob_success2 < 0
print "Unstructured P(hier < ind): ", np.mean(np.mean(tmp,2),0)
print "Structured P(hier < ind): ", np.mean(np.mean(tmp2,2),0)
print "Difference: ", np.mean(np.mean(tmp2,2),0) - np.mean(np.mean(tmp,2),0)
print ''
tmp = diff_prob_success == 0
tmp2 = diff_prob_success2 == 0
print "Unstructured P(hier == ind): ", np.mean(np.mean(tmp,2),0)
print "Structured P(hier == ind): ", np.mean(np.mean(tmp2,2),0)
print "Difference: ", np.mean(np.mean(tmp2,2),0) - np.mean(np.mean(tmp,2),0)
print ''
print "Unstructured frac info: ", np.mean(np.mean(upper_seq_frac_info,2),0)
print "Structured frac info: ", np.mean(np.mean(upper_seq_frac_info2,2),0)
print "Difference: ", np.mean(np.mean(upper_seq_frac_info2,2),0) - np.mean(np.mean(upper_seq_frac_info,2),0)
print ''
print "Unstructured abs info: ", np.mean(np.mean(upper_seq_info,2),0)
print "Structured abs info: ", np.mean(np.mean(upper_seq_info2,2),0)
print "Difference: ", np.mean(np.mean(upper_seq_info2,2),0) - np.mean(np.mean(upper_seq_info,2),0)
print ''









print "Combined Data:"
for ii in range(n_doors):
    print 'Frac cumulative info x success prob', ii, ":", np.corrcoef(diff_prob_success_combined[:,ii,:].flatten(), upper_seq_frac_info_combined[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success_combined.flatten(), upper_seq_frac_info_combined.flatten())[0,1]

for ii in range(n_doors):
    print 'Cumulative info x success prob', ii, ":", np.corrcoef(diff_prob_success_combined[:,ii,:].flatten(), upper_seq_info_combined[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success_combined.flatten(), upper_seq_info_combined.flatten())[0,1]

for ii in range(n_doors-1):
    print 'Frac conditional cum info x success prob', ii+1, ":", np.corrcoef(diff_prob_success_combined[:,ii+1,:].flatten(), upper_seq_cond_frac_info_combined[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success_combined[:,1:,:].flatten(), upper_seq_cond_frac_info_combined.flatten())[0,1]

for ii in range(n_doors-1):
    print 'Conditional cum info x success prob', ii+1, ":", np.corrcoef(diff_prob_success_combined[:,ii+1,:].flatten(), upper_seq_cond_info_combined[:,ii,:].flatten())[0,1]
print np.corrcoef(diff_prob_success_combined[:,1:,:].flatten(), upper_seq_cond_info_combined.flatten())[0,1]
print ''



# t-test

Y = np.array(diff_prob_success_combined.flatten())
Y[Y < 1] = 0
Y = Y.reshape((Y.size,1))
print np.sum(Y)/Y.size, 1-np.sum(Y)/Y.size


# Y2 == 1 - P(ind > hier), that is prob that ind is not better than hier
Y2 = np.array(diff_prob_success_combined.flatten())
Y2[Y2 > -1] = 0
Y2 += 1
Y2 = Y2.reshape((Y2.size,1))



X2 = np.ones(diff_prob_success_combined.shape)
n_unstructured = diff_prob_success.shape[0]
X2[n_unstructured:] = -1
X2 = X2.flatten()
X2 = X2.reshape((X2.size,1))



for ii in range(n_doors):
    diff_prob_success_subset = diff_prob_success_combined[:,ii,:]
    info_subset = upper_seq_frac_info_combined[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Fractional cumulative info, door " + str(ii) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_frac_info_combined[diff_prob_success_combined > 0]
frac_info_miss = upper_seq_frac_info_combined[diff_prob_success_combined < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Fractional cumulative info, all doors:", t_val, p_val
print ''

X = upper_seq_frac_info_combined.flatten()
X = zscore(X)
X = X.reshape((X.size,1))
clf = LogisticRegression(C=1E16).fit(X,Y)
print 'Original regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X,Y)
Y_pred = clf.predict(X)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y,clf)
print 'BIC: ', BIC(X,Y,clf)
print ''


X_tmp = np.random.permutation(X)
X_tmp = X_tmp.reshape((X_tmp.size,1))
clf = LogisticRegression(C=1E16).fit(X_tmp,Y)
print 'Random regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X_tmp,Y)
print 'Likelihood ratio test: ', likelihood_ratio_test(X_tmp,Y,clf)
print 'BIC: ', BIC(X_tmp,Y,clf)
print ''

clf = LogisticRegression(C=1E16).fit(X2,Y)
print 'Structure index regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X2,Y)
Y_pred = clf.predict(X2)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X2,Y,clf)
print 'BIC: ', BIC(X2,Y,clf)
print ''

X3 = np.concatenate([X,X2], axis=1)
clf = LogisticRegression(C=1E16).fit(X3,Y)
print 'Combined regression:'
print "Logistic regression (info coeff, structure coeff, intercept, excess intercept): ", (clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X3,Y)
Y_pred = clf.predict(X3)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X3,Y,clf)
print 'BIC: ', BIC(X3,Y,clf)
print ''


print 'Regression with 1 - P(ind > hier):'
X = upper_seq_frac_info_combined.flatten()
X = zscore(X)
X = X.reshape((X.size,1))
clf = LogisticRegression(C=1E16).fit(X,Y2)
print 'Original regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print 'Score: ', clf.score(X,Y2)
Y_pred = clf.predict(X)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y2,clf)
print 'BIC: ', BIC(X,Y2,clf)
print ''


clf = LogisticRegression(C=1E16).fit(X2,Y2)
print 'Structure index regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print 'Score: ', clf.score(X2,Y2)
Y_pred = clf.predict(X2)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X2,Y2,clf)
print 'BIC: ', BIC(X2,Y2,clf)
print ''

X3 = np.concatenate([X,X2], axis=1)
clf = LogisticRegression(C=1E16).fit(X3,Y2)
print 'Combined regression:'
print "Logistic regression (info coeff, structure coeff, intercept, excess intercept): ", (clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print 'Score: ', clf.score(X3,Y2)
Y_pred = clf.predict(X3)
print 'Score2: ', np.sum(Y_pred==0)/Y.size
print 'Likelihood ratio test: ', likelihood_ratio_test(X3,Y2,clf)
print 'BIC: ', BIC(X3,Y2,clf)



print ''


for ii in range(n_doors):
    diff_prob_success_subset = diff_prob_success_combined[:,ii,:]
    info_subset = upper_seq_info_combined[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Absolute cumulative info, door " + str(ii) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_info_combined[diff_prob_success_combined > 0]
frac_info_miss = upper_seq_info_combined[diff_prob_success_combined < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Absolute cumulative info, all doors:", t_val, p_val

X = upper_seq_info_combined.flatten()
X = zscore(X)
X = X.reshape((X.size,1))
clf = LogisticRegression(C=1E16).fit(X,Y)
print 'Original regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X,Y)
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y,clf)
print 'BIC: ', BIC(X,Y,clf)
print ''

clf = LogisticRegression(C=1E16).fit(X2,Y)
print 'Structure index regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X2,Y)
print 'Likelihood ratio test: ', likelihood_ratio_test(X2,Y,clf)
print 'BIC: ', BIC(X2,Y,clf)
print ''

X3 = np.concatenate([X,X2], axis=1)
clf = LogisticRegression(C=1E16).fit(X3,Y)
print 'Combined regression:'
print "Logistic regression (info coeff, structure coeff, intercept, excess intercept): ", (clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Score: ', clf.score(X3,Y)
print 'Likelihood ratio test: ', likelihood_ratio_test(X3,Y,clf)
print 'BIC: ', BIC(X3,Y,clf)
print ''



print 'Regression with 1 - P(ind > hier):'
X = upper_seq_info_combined.flatten()
X = zscore(X)
X = X.reshape((X.size,1))
clf = LogisticRegression(C=1E16).fit(X,Y2)
print 'Original regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print 'Score: ', clf.score(X,Y2)
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y2,clf)
print 'BIC: ', BIC(X,Y2,clf)
print ''

clf = LogisticRegression(C=1E16).fit(X2,Y2)
print 'Structure index regression:'
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print 'Score: ', clf.score(X2,Y2)
print 'Likelihood ratio test: ', likelihood_ratio_test(X2,Y2,clf)
print 'BIC: ', BIC(X2,Y2,clf)
print ''

X3 = np.concatenate([X,X2], axis=1)
clf = LogisticRegression(C=1E16).fit(X3,Y2)
print 'Combined regression:'
print "Logistic regression (info coeff, structure coeff, intercept, excess intercept): ", (clf.coef_[0][0], clf.coef_[0][1], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print 'Score: ', clf.score(X3,Y2)
print 'Likelihood ratio test: ', likelihood_ratio_test(X3,Y2,clf)
print 'BIC: ', BIC(X3,Y2,clf)
print ''



for ii in range(n_doors-1):
    diff_prob_success_subset = diff_prob_success_combined[:,ii+1,:]
    info_subset = upper_seq_cond_frac_info_combined[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Absolute conditional cumulative info, door " + str(ii+1) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_cond_frac_info_combined[diff_prob_success_combined[:,1:,:] > 0]
frac_info_miss = upper_seq_cond_frac_info_combined[diff_prob_success_combined[:,1:,:] < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Absolute conditional cumulative info, all doors:", t_val, p_val
print ''


for ii in range(n_doors-1):
    diff_prob_success_subset = diff_prob_success_combined[:,ii+1,:]
    info_subset = upper_seq_cond_info_combined[:,ii,:]
    frac_info_hits = info_subset[diff_prob_success_subset > 0]
    frac_info_miss = info_subset[diff_prob_success_subset < 1]
    t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
    print "Fractional conditional cumulative info, door " + str(ii+1) + ": ", t_val, p_val
    
frac_info_hits = upper_seq_cond_info_combined[diff_prob_success_combined[:,1:,:] > 0]
frac_info_miss = upper_seq_cond_info_combined[diff_prob_success_combined[:,1:,:] < 1]
t_val, p_val = ttest(frac_info_hits, frac_info_miss, equal_var=equal_var)
print "Fractional conditional cumulative info, all doors:", t_val, p_val
print ''




raise







# full_upper_seq_info = np.array( [ [ [ mutual_info[ii]['upper sequences cumulative info'][kk][jj] for jj in range(n_rooms) ] for kk in range(n_doors) ] for ii in range(1,n_sims) ] )

# for ii in range(n_doors):
#     print 'Success prob ', ii, ":", np.corrcoef(hier_prob_success[:,ii,1:].flatten(), full_upper_seq_info[:,ii,:-1].flatten())[0,1]




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
hier_steps = hier_steps[hier_steps['Iteration'].isin(range(1,n_sims))]
iterations = np.array(hier_steps['Iteration'])
indep_steps = indep_steps[indep_steps['Iteration'].isin(range(1,n_sims))]


indep_steps = indep_steps.sort_values(by='Iteration')
hier_steps = hier_steps.sort_values(by='Iteration')

indep_steps = np.array(list(indep_steps['Cumulative Steps']))
hier_steps = np.array(list(hier_steps['Cumulative Steps']))

# iterations /= 2




# sublvl_info = sublvl_info[iterations]
# upper_info = upper_info[iterations]
# upper_x_sublvl_info = upper_x_sublvl_info[iterations]
# same_sublvl_info = same_sublvl_info[:,iterations]
# same_sublvl_avg_info = same_sublvl_avg_info[iterations]
# hier_door_info = hier_door_info[:,iterations]
# hier_door_avg_info = hier_door_avg_info[iterations]
# flat_subgoal_info = flat_subgoal_info[iterations]
# flat_door_info = flat_door_info[:,iterations]
# flat_door_info_avg = flat_door_info_avg[iterations]
# upper_seq_info = upper_seq_info[:,iterations]




print 'Success prob x hier steps:', np.corrcoef(np.mean(np.mean(hier_prob_success,axis=2),axis=1), hier_steps)[0,1]
print 'Success prob x hier steps:', np.corrcoef(np.mean(hier_prob_success[:,1,:],axis=1), hier_steps)[0,1]
print 'Success prob x hier steps:', np.corrcoef(np.mean(diff_prob_success[:,3,:],axis=1), indep_steps-hier_steps)[0,1]
print 'Success prob x hier steps:', np.corrcoef(np.mean(np.mean(diff_prob_success,axis=2),axis=1), indep_steps-hier_steps)[0,1]
# raise




# sublvl_info2 = np.array([ mutual_info2[ii]['sublvl'][0][-1] for ii in range(nsims) ])
# upper_info2 = np.array([ mutual_info2[ii]['upper room'][0][-1] for ii in range(nsims) ])
# upper_x_sublvl_info2 = np.array([ mutual_info2[ii]['upper mapping x sublvl rewards'][0][-1] for ii in range(nsims) ])

# same_sublvl_info2 = np.array( [ [ mutual_info2[ii]['same sublvl mapping x goal'][kk][-1] for ii in range(nsims) ] for kk in range(n_sublvls) ] )
# same_sublvl_avg_info2 = np.mean( same_sublvl_info2, axis=0 ) 

# hier_door_info2 = np.array( [ [ mutual_info2[ii]['upper mapping x individual door'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )
# hier_door_avg_info2 = np.mean( hier_door_info2, axis=0 ) 

# flat_subgoal_info2 = np.array([ mutual_info2[ii]['mappings x sublvl goal'][0][-1] for ii in range(nsims) ])
# flat_door_info2 = np.array( [ [ mutual_info2[ii]['mappings x individual door'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )
# flat_door_info_avg2 = np.mean(flat_door_info2, axis=0)

# upper_seq_info2 = np.array( [ [ mutual_info2[ii]['upper sequences'][kk][-1] for ii in range(nsims) ] for kk in range(n_doors) ] )





# X0 = results2[results2['Success'] == True]
# indep_steps2 = X0[X0['Model']=='Independent']
# hier_steps2 = X0[X0['Model']=='Hierarchical']

# iterations = np.array(indep_steps2['Iteration'])
# hier_steps2 = hier_steps2[hier_steps2['Iteration'].isin(iterations)]
# iterations = np.array(hier_steps2['Iteration'])
# indep_steps2 = indep_steps2[indep_steps2['Iteration'].isin(iterations)]


# indep_steps2 = indep_steps2.sort_values(by='Iteration')
# hier_steps2 = hier_steps2.sort_values(by='Iteration')

# indep_steps2 = np.array(list(indep_steps2['Cumulative Steps']))
# hier_steps2 = np.array(list(hier_steps2['Cumulative Steps']))


# iterations /= 2


# sublvl_info2 = sublvl_info2[iterations]
# upper_info2 = upper_info2[iterations]
# upper_x_sublvl_info2 = upper_x_sublvl_info2[iterations]
# same_sublvl_info2 = same_sublvl_info2[:,iterations]
# same_sublvl_avg_info2 = same_sublvl_avg_info2[iterations]
# hier_door_info2 = hier_door_info2[:,iterations]
# hier_door_avg_info2 = hier_door_avg_info2[iterations]
# flat_subgoal_info2 = flat_subgoal_info2[iterations]
# flat_door_info2 = flat_door_info2[:,iterations]
# flat_door_info_avg2 = flat_door_info_avg2[iterations]
# upper_seq_info2 = upper_seq_info2[:,iterations]






# sublvl_info = np.concatenate((sublvl_info,sublvl_info2), axis=0)
# upper_info = np.concatenate((upper_info,upper_info2), axis=0)
# upper_x_sublvl_info = np.concatenate((upper_x_sublvl_info,upper_x_sublvl_info2), axis=0)
# same_sublvl_info = np.concatenate((same_sublvl_info,same_sublvl_info2), axis=1)
# same_sublvl_avg_info = np.concatenate((same_sublvl_avg_info,same_sublvl_avg_info2), axis=0)
# hier_door_info = np.concatenate((hier_door_info,hier_door_info2), axis=1)
# hier_door_avg_info = np.concatenate((hier_door_avg_info,hier_door_avg_info2), axis=0)
# flat_subgoal_info = np.concatenate((flat_subgoal_info,flat_subgoal_info2), axis=0)
# flat_door_info = np.concatenate((flat_door_info,flat_door_info2), axis=1)
# flat_door_info_avg = np.concatenate((flat_door_info_avg,flat_door_info_avg2), axis=0)
# upper_seq_info = np.concatenate((upper_seq_info,upper_seq_info2), axis=1)



# indep_steps = np.concatenate((indep_steps,indep_steps2),axis=0)
# hier_steps = np.concatenate((hier_steps,hier_steps2),axis=0)



# indep_hier = indep_steps-hier_steps


# print 'Upper corr: ', np.corrcoef(upper_info, indep_hier)[0,1]
# print 'Sublvl corr: ', np.corrcoef(sublvl_info, indep_hier)[0,1]
# print 'Upper x sublvl corr: ', np.corrcoef(upper_x_sublvl_info, indep_hier)[0,1]
# print ''

# print 'Upper sequence x door 1: ', np.corrcoef(upper_seq_info[0], indep_hier)[0,1]
# print 'Upper sequence x door 2: ', np.corrcoef(upper_seq_info[1], indep_hier)[0,1]
# print 'Upper sequence x door 3: ', np.corrcoef(upper_seq_info[2], indep_hier)[0,1]
# print 'Upper sequence x door 4: ', np.corrcoef(upper_seq_info[3], indep_hier)[0,1]



# print ''
# print 'Upper mapping x door 1: ', np.corrcoef(hier_door_info[0], indep_hier)[0,1]
# print 'Upper mapping x door 2: ', np.corrcoef(hier_door_info[1], indep_hier)[0,1]
# print 'Upper mapping x door 3: ', np.corrcoef(hier_door_info[2], indep_hier)[0,1]
# print 'Upper mapping x door 4: ', np.corrcoef(hier_door_info[3], indep_hier)[0,1]
# print 'Upper mapping x door avg: ', np.corrcoef(hier_door_avg_info, indep_hier)[0,1]

# print ''
# print 'Flat mappings x door 1: ', np.corrcoef(flat_door_info[0], indep_hier)[0,1]
# print 'Flat mappings x door 2: ', np.corrcoef(flat_door_info[1], indep_hier)[0,1]
# print 'Flat mappings x door 3: ', np.corrcoef(flat_door_info[2], indep_hier)[0,1]
# print 'Flat mappings x door 4: ', np.corrcoef(flat_door_info[3], indep_hier)[0,1]
# print 'Flat mappings x door avg: ', np.corrcoef(flat_door_info_avg, indep_hier)[0,1]

# print ''
# print 'Diff info - mappings x door 1: ', np.corrcoef(hier_door_info[0] - flat_door_info[0], indep_hier)[0,1]
# print 'Diff info - mappings x door 2: ', np.corrcoef(hier_door_info[1] - flat_door_info[1], indep_hier)[0,1]
# print 'Diff info - mappings x door 3: ', np.corrcoef(hier_door_info[2] - flat_door_info[2], indep_hier)[0,1]
# print 'Diff info - mappings x door 4: ', np.corrcoef(hier_door_info[3] - flat_door_info[3], indep_hier)[0,1]
# print 'Diff info - mappings x door avg: ', np.corrcoef(hier_door_avg_info - flat_door_info_avg, indep_hier)[0,1]

# print ''
# print 'Same sublvl mapping x subgoal 1: ', np.corrcoef(same_sublvl_info[0], indep_hier)[0,1]
# print 'Same sublvl mapping x subgoal 2: ', np.corrcoef(same_sublvl_info[1], indep_hier)[0,1]
# print 'Same sublvl mapping x subgoal 3: ', np.corrcoef(same_sublvl_info[2], indep_hier)[0,1]
# print 'Same sublvl mapping x subgoal avg: ', np.corrcoef(same_sublvl_avg_info, indep_hier)[0,1]

# print ''
# print 'Flat mapping x flat subgoals: ', np.corrcoef(flat_subgoal_info, indep_hier)[0,1]

# print ''
# print 'Diff mappings x subgoal 1: ', np.corrcoef(same_sublvl_info[0] - flat_subgoal_info, indep_hier)[0,1]
# print 'Diff mappings x subgoal 2: ', np.corrcoef(same_sublvl_info[1] - flat_subgoal_info, indep_hier)[0,1]
# print 'Diff mappings x subgoal 3: ', np.corrcoef(same_sublvl_info[2] - flat_subgoal_info, indep_hier)[0,1]
# print 'Diff mappings x subgoals: ', np.corrcoef(same_sublvl_avg_info - flat_subgoal_info, indep_hier)[0,1]
