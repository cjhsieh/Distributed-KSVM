#!/bin/bash                         # Use Bash Shell
#SBATCH -J run_svm                  # Job Name
#SBATCH -o output_file              # Name of the output file (eg. myMPI.oJobID)
#SBATCH -N 32  
#SBATCH -n 32
#SBATCH -t 1:00:00       # Run time (hh:mm:ss) - 1.5 hours

export OMP_NUM_THREADS=20; 
ibrun tacc_affinity ./svm-train-mpi -m 20000 -c 32 -g 32 -T 200000 -R 1 -D 0 -e 0.5 ../data/covtype_train ../data/covtype_test

