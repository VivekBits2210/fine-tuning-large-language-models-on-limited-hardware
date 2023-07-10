# fine-tuning-large-language-models-on-limited-hardware
NYU Tandon, ECE-GY 9143: High Performance Machine Learning, End Semester Project

## Project:
Optimize the process of domain adaptation in natural language processing, i.e., fine-tuning large language models for a particular domain on limited hardware . This is done by using 8-bit quantization, LoRA and other techniques.

## Repository:
- Fine_Tuning.ipynb: Main notebook for training and evaluation
- train_v4.py: Same script as ipynb in python script form, for submitting a sbatch job
- Inference.ipynb: Inference notebook, for generating text from a saved model
- shell scripts/data_download.sh: Script for downloading data
- shell scripts/run_train_job.sh: Script for running the train job (automatically runs train_v4.py)
- gpt2_logs: Training and validation logs for the GPT-2 fine tuning run
- opt_logs:  Training and validation logs for the OPT fine tuning run
- sbatch_job_log: Logs from the sbatch job for GPT-2 fine tuning run

## How to run the code:
- Create sbatch job and run run_train_job.sh (16 cores, 60 GB RAM, 1 RTX GPU)

## Results:
Achieved a perplexity score of 7.24 while fine-tuning GPT-2 model (1.5B param model) on the NiH grants dataset after 17 epochs. Qualitatively, the text generated is plausibly similar to an NIH grant.
