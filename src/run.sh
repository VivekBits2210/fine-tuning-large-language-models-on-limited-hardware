#!/bin/bash -e
#SBATCH --output=%x_%j.txt --time=47:55:00 --wrap "sleep infinity" --cpus-per-task=8  --gres=gpu:rtx8000:1
echo "Hostname: $(hostname)"
echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep 'Mem:' | awk '{print $4}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi | grep MiB |  awk '{print $9 $10 $11}')"

module purge
module load intel/19.1.2
module load python/intel/3.8.6
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
cd /scratch/vgn2004/src

# Default values
NET_ID=${NET_ID:-"vpn2004"}
ENV=${ENV:-"pre_prod"}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_TOKENS=${MAX_TOKENS:-512}
MIN_GENERATION=${MIN_GENERATION:-128}
MODEL_NAME=${MODEL_NAME:-"facebook/opt-125m"}
DATASET_NAME=${DATASET_NAME:-"NIH_ExPORTER_awarded_grant_text"}
BATCH_SIZE=${BATCH_SIZE:-64}

python runner.py --net-id "$NET_ID" --env "$ENV" --num-workers "$NUM_WORKERS" --max-tokens "$MAX_TOKENS" --min-generation "$MIN_GENERATION" --model-name "$MODEL_NAME" --dataset-name "$DATASET_NAME" --batch-size "$BATCH_SIZE"


#sbatch --job-name=your_job_name --export=NET_ID="vgn2004",ENV="post_prod",BATCH_SIZE=32 run.sh