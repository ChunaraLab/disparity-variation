#!/bin/bash

#SBATCH --job-name=dervariability
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-10
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --output=dervariability-%A_%a.out
#SBATCH --mail-type=end
#SBATCH --mail-user=hs3673@nyu.edu

module purge

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif
OVERLAY_FILE=/scratch/hs3673/singularity/singularity-der.ext3

singularity exec --nv \
        --overlay $OVERLAY_FILE:ro $SINGULARITY_IMAGE\
        /bin/bash -c "source /ext3/env.sh; bash run.sh $SLURM_ARRAY_TASK_ID"
