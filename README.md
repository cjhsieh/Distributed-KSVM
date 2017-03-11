Distributed Block Minimization for Nonlinear Kernel SVM (PBM-SVM)
=================================================================

PBM-SVM is a distributed version of LIBSVM using a distributed greedy
coordinate descent algorithm. Please note that the current version
only supports binary classification (with label +1 and -1). 
For more details about this algorithm please refer to the following paper:

```
Communication-Efficient Distributed Block Minimization forNonlinear Kernel Machines
Si Si, Cho-Jui Hsieh, and Inderjit S. Dhillon, 2017. 
```

Build
---------------

To build the program, simply run `make`. Note that you might need to modify
the compiler varibles in the `Makefile`: 

CXX = icc 
MPICXX = mpicxx

CXX is the compiler (GCC/G++), where we do not need the c++11 support. 
MPICXX is the mpi compiler. 
For example, you may specify CXX=g++ and MPICXX=mpicxx.mpich2

Two binaries, `svm-train-mpi` (for distributed training) and `svm-predict` 
(for predicting with a single machine) will be built.  

Data Preparation 
----------------

We have included the covtype dataset: 

covtype: `covtype_train` is training data and `covtype_test` is testing data

We support the datasets in the LIBSVM format (where we only consider binary classifcation
with labels +1/-1). You can download other datasets from LIBSVM datasets
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html 

Usage
----------------

```
./svm-train-mpi [options] training_set_file test_set_file 
options:
-g gamma : set gamma in kernel function (default 1/num_features)
-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-m cachesize : set cache memory size in MB (default 100)
-e epsilon : set tolerance of termination criterion (default 0.1)
-T T : set the maximum number of outer iterations (default 10)
-A A : set the solver type (0: our method(default), 1: SGD)
-R R : set the cluster (0: kmeans(default), 1:random)
-F F : set the function type (0: SVM(default), 1:logistic regression)
-N N : set the number of threads per machine (default: environment variable OMP_NUM_THREADS)
-D D : D=0: not using divide-and-conquer (default), D=1: using divide-and-conquer
-p p : print out the accuracy/objective function every p seconds (default p=10)
```

Examples:
----------
See the following files for running the experiments with sbatch: 

`go_cov_multicore_random.sh`: Run the covtype dataset with 32 machines (each 20 cores) with random partition. 

`go_cov_multicore.sh`: Run the covtype dataset with 32 machines (each 20 cores) with kmeans partition.

`go_cov_multicore_dc.sh`: Run the covtype dataset with 32 machines (each 20 cores) with kmeans partition and divide-and-conquer strategy.


Additional Information
----------------------

If your have any questions or comments, please open an issue on Github,
or send an email to chohsieh@ucdavis.edu. We appreciate your feedback.

