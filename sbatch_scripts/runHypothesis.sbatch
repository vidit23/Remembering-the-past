#!/bin/bash
#
#SBATCH --job-name=doesntmatter
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=40GB
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source ~/.bashrc # source your bashrc to get access to conda.

cd /home/vvb238/NLP/transformers/examples/question-answering/
echo "in proper CWD"

conda activate nlp # activate your conda env with all your libraries ! #Important.
echo "Activate conda env"

model_path="bert-base-uncased"

seperatedTraining=$(echo $trainingDatasets | tr ";" "\n")

for dataset in $seperatedTraining
do
  echo "---------------------------- TRAINING ----------------------------"
  echo ${dataset}
  echo ${model_path}
  python run_squad.py \
    --model_type bert \
    --model_name_or_path $model_path \
    --do_train \
    --do_lower_case \
    --data_dir "/home/vvb238/NLP/sq_datasets" \
    --dataset_name ${dataset} \
    --train_file "train/${dataset}_squad_format.json" \
    --per_gpu_train_batch_size 6 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir "/home/vvb238/NLP/run_outputs/${SLURM_JOB_NAME}" \
    --overwrite_output_dir \
    --save_steps 10000 \
    --logging_steps 1000 \
    --cache_dir "/home/vvb238/NLP/run_caches/${SLURM_JOB_NAME}"

  model_path="/home/vvb238/NLP/run_outputs/${SLURM_JOB_NAME}"
done


model_path="/home/vvb238/NLP/run_outputs/${SLURM_JOB_NAME}"

seperatedTesting=$(echo $testingDatasets | tr ";" "\n")

for dataset in $seperatedTesting
do
  echo "---------------------------- TESTING ----------------------------"
  echo "${dataset}_dev"
  echo ${model_path}
  python run_squad.py \
    --model_type bert \
    --model_name_or_path $model_path \
    --do_eval \
    --do_lower_case \
    --data_dir "/home/vvb238/NLP/sq_datasets" \
    --dataset_name ${dataset} \
    --predict_file "dev/${dataset}_squad_format.json" \
    --per_gpu_eval_batch_size 6 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir "/home/vvb238/NLP/run_outputs/${SLURM_JOB_NAME}" \
    --cache_dir "/home/vvb238/NLP/run_caches/${SLURM_JOB_NAME}"

done