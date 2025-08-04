#!/bin/bash
#SBATCH --array=1-10%3
#SBATCH --job-name=fno4fick
#SBATCH --nodes=1                # nombre de noeuds
#SBATCH --ntasks=1               # nombre total de tâches sur tous les nœuds
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=hs_slurm/dcv_hist/out/slurm-%A_%a.txt
#SBATCH --error=hs_slurm/dcv_hist/err/slurm-%A_%a.txt
#SBATCH --mail-type=ALL
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# export TMPDIR=/scratch/<project>/tmp

BATCH_HIST="batch.txt"

NAMES=("1_final")

CMD=$"srun --unbuffered python -u -m experiments.train.main -l 4 -m 64 -c 64 -e 200 -d examples/no_limit/Nt=1000_Nr=200/train/ -n ${NAMES} -b 16"

echo "start"
ml python/3.12
ml cuda/12.4
source ../env/bin/activate 
export PYTHONPATH=$PWD 
echo "$SLURM_ARRAY_TASK_ID|$CMD" >> $BATCH_HIST
$CMD
deactivate
echo "end"