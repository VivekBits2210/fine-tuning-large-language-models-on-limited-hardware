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
cd /scratch/vgn2004
venv_name="finetuning_venv"
if [ ! -d "$venv_name" ]; then
  python -m venv "$venv_name"
fi
source ./$venv_name/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install  --use-feature=2020-resolver Cython zstandard datasets psutil transformers torch bitsandbytes accelerate loralib peft pynvml deepspeed
python3 ./train_v4.py
