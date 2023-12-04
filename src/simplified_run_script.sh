#!/bin/bash -e
#SBATCH --job-name=trainer --output=sbatch_output.txt --time=23:55:00 --wrap "sleep infinity" --cpus-per-task=8 --mem=64G  --gres=gpu:rtx8000:2
echo "Hostname: $(hostname)"
echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep 'Mem:' | awk '{print $4}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GPU Memory: $(nvidia-smi | grep MiB |  awk '{print $9 $10 $11}')"
nvidia-smi
module purge
module load intel/19.1.2
module load python/intel/3.8.6
module load cuda/11.6.2
module load cudnn/8.6.0.163-cuda11
cd /scratch/vgn2004
python -m pip install --upgrade pip setuptools
venv_name="qlora_latest_venv"
if [ ! -d "$venv_name" ]; then
  python -m venv "$venv_name"
fi
source ./$venv_name/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install --use-feature=2020-resolver datasets transformers torch bitsandbytes accelerate peft pynvml scikit-learn wandb numpy scipy pandas psutil deepspeed
python3 /scratch/vgn2004/fine-tuning-large-language-models-on-limited-hardware/src/simplified_qlora_replication.py --is_quantized True --experiment_name sbatch_first_try
