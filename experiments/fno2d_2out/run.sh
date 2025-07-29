#!/bin/bash
#SBATCH --array=1%1
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

CMD=$"srun python -m experiments.fno2d_2out.main -l 4 -m 64 -c 64 -e 200 -d examples/no_limit/Nt=1000_Nr=200/train/ -n no_limit_2out_Nt=1000_Nr=200_without_batch_hist_problem_processing_print -b 16 -r True >> processing.txt"

echo "start"
ml python/3.12
ml cuda/12.4
source ../env/bin/activate 
export PYTHONPATH=$PWD 
echo "$SLURM_ARRAY_TASK_ID|$CMD" >> $BATCH_HIST
$CMD
deactivate
echo "end"