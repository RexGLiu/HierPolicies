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



# t-test and logistic regression
equal_var = False

Y = np.array(diff_prob_success.flatten())
Y[Y < 1] = 0
Y = Y.reshape((Y.size,1))



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
clf = LogisticRegression(C=1E25, tol=1e-15, max_iter=100000).fit(X,Y)
print "Logistic regression (coefficient, intercept, excess intercept): ", (clf.coef_[0][0], clf.intercept_[0], clf.intercept_[0] - intercept_baseline)
print "Intercept to equivalent baseline: ", intercept_to_baseline(clf.intercept_[0])
print "Const model intercept: ", -np.log(Y.size/np.sum(Y) - 1)
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y,clf)
print 'BIC: ', BIC(X,Y,clf)
print ''



X_test = np.arange(-2.5, 1, 0.1)
X_test = X_test.reshape((-1,1))
plt.figure(figsize=(5, 4.5))
plt.scatter(X, Y, marker='.')
plt.plot(X_test, clf.predict_proba(X_test)[:,1], 'navy')
plt.xlabel("Z-scored fractional cumulative info")
plt.ylabel("Prob[hier > indep]")
plt.savefig("figs/HierarchicalRooms_indep_no_structure_Prob_vs_Info.png", dpi=300, bbox_inches='tight')
plt.show()


sort_idx = np.argsort(X.flatten())
X_sorted = X[sort_idx]
Y_sorted = Y[sort_idx]


bar_width = 0.4
bins = np.arange(-2.5,0.7+bar_width,bar_width)
bins = bins.reshape((1,-1))

bar_centres = (bins[0,1:] + bins[0,:-1])/2

X_idx = np.logical_and(X_sorted < bins[0,1:], X_sorted > bins[0,:-1])
Y_frac = np.array([np.sum(Y_sorted[X_idx[:,ii]])/X_idx[:,ii].size for ii in range(bins.size - 1)])


X_test = np.arange(-2.5, 1, 0.1)
X_test = X_test.reshape((-1,1))
plt.figure(figsize=(5, 4.5))
plt.bar(bar_centres, Y_frac, width = bar_width)
plt.plot(X_test, clf.predict_proba(X_test)[:,1], 'navy')
plt.xlabel("Z-scored fractional cumulative info")
plt.ylabel("Prob[hier > indep]")
plt.savefig("figs/HierarchicalRooms_indep_no_structure_Counts_vs_Info.png", dpi=300, bbox_inches='tight')
plt.show()



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
print 'Likelihood ratio test: ', likelihood_ratio_test(X,Y,clf)
print 'BIC: ', BIC(X,Y,clf)
print ''


# info content conditioned on the mapping

print 'Info content conditioned on the mapping:'
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


