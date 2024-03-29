import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec
from tqdm import tqdm

from model import make_task, JointClustering, IndependentClusterAgent, FlatControlAgent, MetaAgent, HierarchicalAgent

# Define a function to Simulate the Models
def simulate_one(agent_class, simulation_number, task, agent_kwargs=None, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    if agent_kwargs is not None:
        agent = agent_class(task, **agent_kwargs)
    else:
        agent = agent_class(task)

    results, clusterings = agent.generate()
    results['Simulation Number'] = [simulation_number] * len(results)

    return results, clusterings


def simulate_mixed_task(n_sim, task_kwargs, agent_kwargs=None, alpha=2.0, pruning_threshold=1000,
                        evaluate=False, seed=None,
                        meta_kwargs=None, metarl_kwargs=None):
    if agent_kwargs is None:
        agent_kwargs = dict(alpha=alpha)
    elif 'alpha' not in agent_kwargs.keys():
        agent_kwargs['alpha'] = alpha
        
    hier_agent_kwargs = { k:v for k,v in agent_kwargs.iteritems() if k != 'alpha' }
        
    if seed is not None:
        np.random.seed(seed)

    if meta_kwargs is None:
        meta_kwargs = dict(m_biases=[0.0, 0.0])
    for k, v in agent_kwargs.iteritems():
        meta_kwargs[k] = v

    if metarl_kwargs is None:
        metarl_kwargs = dict(m_biases=[0.0, 0.0])
    for k, v in agent_kwargs.iteritems():
        metarl_kwargs[k] = v

    results_jc = [None] * n_sim
    results_ic = [None] * n_sim
    results_fl = [None] * n_sim
    results_mx = [None] * n_sim
    results_h = [None] * n_sim
    
    task_list = [make_task(**task_kwargs) for _ in range(n_sim)]

    for ii in tqdm(range(n_sim)):
        task = task_list[ii]
        task.reset()
        results_jc[ii], _ = simulate_one(JointClustering, ii, task, agent_kwargs=agent_kwargs)

    for ii in tqdm(range(n_sim)):
        task = task_list[ii]
        task.reset()
        results_ic[ii], _ = simulate_one(IndependentClusterAgent, ii, task, agent_kwargs=agent_kwargs)

    for ii in tqdm(range(n_sim)):
        task = task_list[ii]
        task.reset()
        results_fl[ii], _ = simulate_one(FlatControlAgent, ii, task, agent_kwargs=agent_kwargs)

    for ii in tqdm(range(n_sim)):
        task = task_list[ii]
        task.reset()
        results_h[ii], _ = simulate_one(HierarchicalAgent, ii, task, agent_kwargs=hier_agent_kwargs)
        
    for ii in tqdm(range(n_sim)):
        task = task_list[ii]
        task.reset()
        results_mx[ii], _ = simulate_one(MetaAgent, ii, task, agent_kwargs=meta_kwargs)

    results_jc = pd.concat(results_jc)
    results_ic = pd.concat(results_ic)
    results_fl = pd.concat(results_fl)
    results_mx = pd.concat(results_mx)
    results_h = pd.concat(results_h)


    results_jc['Model'] = ['Joint'] * len(results_jc)
    results_ic['Model'] = ['Independent'] * len(results_ic)
    results_fl['Model'] = ['Flat'] * len(results_fl)
    results_mx['Model'] = ['Meta'] * len(results_mx)
    results_h['Model'] = ['Hierarchical'] * len(results_h)

    return pd.concat([results_jc, results_ic, results_fl, results_mx, results_h])


def plot_results(df, figsize=(6, 3), sharey=True, name=None):
    with sns.axes_style('ticks'):

        _ = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.4)

        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        # define the parameters to plot the results
        # cc = sns.color_palette("Set 2")
        df0 = df[df['In goal']].groupby(['Model', 'Simulation Number', 'Trial Number']).mean()
        df0 = df0.groupby(level=[0, 1]).cumsum().reset_index()
        df0 = df0.rename(index=str, columns={'n actions taken': "Cumulative Steps Taken"})

        tsplot_kwargs = dict(
            time='Trial Number',
            value='Cumulative Steps Taken',
            data=df0,
            unit='Simulation Number',
            condition='Model',
            estimator=np.mean,
            ax=ax0,
            color="Set2",
        )

        sns.tsplot(**tsplot_kwargs)
        df0 = df[df['In goal']].groupby(['Model', 'Simulation Number']).sum()
        cum_steps = [df0.loc[m]['n actions taken'].values for m in set(df.Model)]
        model = []
        for m in set(df.Model):
            model += [m] * int(df[df.Model == m]['Simulation Number'].max() + 1)
        df1 = pd.DataFrame({
                'Cumulative Steps Taken': np.concatenate(cum_steps),
                'Model': model
            })

        sns.violinplot(data=df1, x='Model', y='Cumulative Steps Taken', ax=ax1, palette='Set2',
                       order=["Flat", "Independent", "Joint", "Hierarchical", "Meta"]
                       )
        ax1.set_xticklabels(['Flat', 'Indep.', 'Joint', 'Hier.', 'Meta'])

        _, ub = ax1.get_ylim()
        ax1.set_ylim([0, ub])
        if sharey is True:
            _, ub = ax0.get_ylim()
            ax0.set_ylim([0, ub])


        sns.despine(offset=5)
        
        if name is not None:
            plt.savefig(name, dpi=300, bbox_inches='tight')
