#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=0:30:00
#SBATCH --job-name="nlu_hw4"
#SBATCH --mail-user=osa209@nyu.edu

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=osa209
source activate /home/${NETID}/nlu/env

cd /home/${NETID}/nlu/code/transformers
export PYTHONPATH=/home/${NETID}/nlu/code/transformers/src:$PYTHONPATH
TASK_NAME=boolq
python ./examples/run_glue.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --data_dir /home/${NETID}/BoolQ \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /home/${NETID}/${TASK_NAME}-run/
