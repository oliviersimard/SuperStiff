#!/bin/bash
#SBATCH --job-name=U8_b24.5_mu5.65_fs
#SBATCH --time=35:00:00
#SBATCH --account=def-tremblay
#SBATCH --ntasks=48
#SBATCH --mem-per-cpu=2500MB

cd $SLURM_SUBMIT_DIR
ppnMP2=48

#module reset
#module load nixpkgs/16.09 gcc/5.4.0 boost/1.65.1 python27-mpi4py/2.0.0 
module reset && module load nixpkgs/16.09 gcc/5.4.0 boost/1.65.1 python/2.7.14 mpi4py/3.0.0 scipy-stack
#module add intel64/13.1.3.192 openmpi_intel64/1.6.5 boost64/1.51.0 json_spirit/4.08 python64/2.7.5 cuba/4.0
python ../../../../../../SelfConsistency.py --mpi -np $ppnMP2 
