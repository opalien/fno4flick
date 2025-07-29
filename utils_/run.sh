#!/bin/bash
#SBATCH --array=1-100%100
#SBATCH --job-name=dataset
#SBATCH --nodes=1                # nombre de noeuds
#SBATCH --ntasks=1               # nombre total de tâches sur tous les nœuds
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --output=hs_slurm/dcv_hist/out/slurm-%A_%a.txt
#SBATCH --error=hs_slurm/dcv_hist/err/slurm-%A_%a.txt
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --partition=besteffort

# export TMPDIR=/scratch/<project>/tmp


BATCH_HIST="batch.txt"



CMD=$"srun python -m utils_.database -f examples/no_limit/Nt=1000_Nr=200/ -n 100 -m True -t 0.0 -d 0.0 -N 1000 -T 200"

echo "start"
ml python/3.12
ml conda
conda activate /public/conda/user_envs/thibault.schneeberger/envs/fenicsx-0.9
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD 
echo "$SLURM_ARRAY_TASK_ID|$CMD" >> $BATCH_HIST
$CMD
conda deactivate
echo "end"