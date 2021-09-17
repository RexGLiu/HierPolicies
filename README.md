# Installation Instructions
This codebase is written in Python 2.7 and requires Cython compilation prior to use. Cython requires a C compiler (gcc). If python 2.7, gcc, and pip are installed on your system, all python required dependencies can be installed with
 ```pip install -r requirements.txt ```
 
Each folder has a `setup.py` file for cython compilation. To compile the Cython code in each folder, enter the relevant folder and run
 ```python setup.py build_ext --inplace```


# Folders
The code is split into three folders according to the type of task. `Flat Task` contains the code for all non-hierarchical tasks. `Diabolical HierRooms` contains the code for all diabolical versions of the hierarchical rooms task, and `Non-diabolical HierRooms` contains the code for the non-diabolical versions. Each folder contains its own version of the following scripts and libraries:
 - ```indep_env_run.py``` runs the version of the task with independent statistics.
 - ```joint_env_run.py``` runs the version of the task with joint statistics.
 - ```ambig_env_run.py``` runs the version of the task with ambiguous statistics.
 - each of these files have a ```*_par.py``` counterpart that runs a parallelised version of the script with `mpi4py`. These scripts are intended to be run on a cluster and were the ones used to generate the published data.
 - ```setup.py``` for Cython compliation (see **Installation Instructions**).
 - the library folder ```model``` contains the code for generating the environments as well as the clustering agents (e.g. independent, joint, hierarchical, meta, flat, etc) that run in them.
 - several miscellaneous scripts for analysing or plotting the generated data (see below for details).

## Analysis scripts
Each of the task scripts above will save its results in a set of `.pkl` files. The following scripts are intended to be run on these outputs and will generate the analyses and plots seen in the paper.

### Flat Task
 - ```plot_results_joint_indep.py``` generates a series of plots for the tasks with joint and independent statistics.
 - ```plot_results_ambig_env.py``` generates a series of plots for the task with mixed statistics.

### Diabolical HierRooms
 - ```plot_results_paper_separate.py``` generates bar and histogram plots showing either the average total steps taken or the distribution over the total steps taken in the independent, joint, and mixed statistics version on this task.
 - ```plot_results_paper_doors.py``` generates a series of bar plots showing the probability of successful first visits for the doors and sublevel goals (i.e. the probability of successfully guessing the corrrect door/subgoal on the first attempt when visiting a level/sublevel for the first time).
 - ```task_info_analysis_paper.py``` runs several statistical tests showing the correlation between info content in door sequences and probability of the hierarchical agent being more successful than the independent one in a first-visit to a level/sublevel. It also generates a plot showing this dependency between the probability and info content.

### Non-diabolical HierRooms
 - ```plot_results_paper_separate.py``` generates plots for the analyses of all non-diabolical tasks.
