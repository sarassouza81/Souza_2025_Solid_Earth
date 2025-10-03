#!/bin/bash -v
#SBATCH --job-name=MR_va_30_15_ris0.6_hkoff_1350 # Job name
#SBATCH --mail-type=BEGIN,END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sara_ssouza@usp.br # Where to send mail
#SBATCH --nodes=1 # Run all processes on a single node	
#SBATCH --ntasks=1 # Run a single task		
#SBATCH --cpus-per-task=16 # Number of CPU cores per task
#SBATCH --time=8- # DD-HH:MM:SS
#SBATCH --output=MR_va_30_15_ris0.6_hkoff_1350.log # Standard output and error log

echo $SLURM_JOB_ID # ID of job allocation
echo $SLURM_SUBMIT_DIR # Directory job where was submitted
echo $SLURM_JOB_NODELIST # File containing allocated hostnames
echo $SLURM_NTASKS # Total number of cores for job

module unload openmpi

#run the application:
PETSC_DIR='/temporario2/9879553/petsc'
PETSC_ARCH='v3.15.5-optimized'
MANDYOC='/temporario2/9879553/mandyoc/bin/mandyoc'
MANDYOC_OPTIONS='-seed 0,1,2 -strain_seed 0.0,0.0,1.0'
${PETSC_DIR}/${PETSC_ARCH}/bin/mpirun -n 16 ${MANDYOC} ${MANDYOC_OPTIONS}
