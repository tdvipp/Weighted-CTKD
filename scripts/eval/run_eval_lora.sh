#!/bin/bash
GPUS=(0)
WORK_DIR=/workspace/DSKD
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")
# echo ${DEVICE}

CKPT_PATH=${1}
MODEL_PATH=${2}
BATCH_SIZE=${3-16}

for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} dolly ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} sinst/11_ ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} self-inst ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} dialogsum ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} vicuna ${BATCH_SIZE} $seed
done
