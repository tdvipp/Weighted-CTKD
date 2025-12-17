#! /bin/bash
GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=/workspace/DSKD
CKPT_TYPE="gpt2"
CKPT_NAME="gpt2-base"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_TYPE}/${CKPT_NAME}"
# we use qwen-1.8b as the teacher with the different vocabulary from gpt2
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen1.5-1.8B"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/qwen/Qwen1.5-1.8B"
# data
DATA_DIR="${BASE_PATH}/data/dolly/"
# task
TASK="wctkd"
# hp
BATCH_SIZE=4
LR=0.0005
GRAD_ACC=2
EVAL_BATCH_SIZE=32
EPOCH=20
KD_RATE=0.5
KD_TEMP=2.0
WCTKD_ALPHA=0.5
WCTKD_BETA=0.2
WCTKD_GAMMA=0.3
WCTKD_HIDDEN_GAMMA=0.5
WCTKD_TOP_K=8
# distiller
PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001
M_GLOBAL_PATH="${BASE_PATH}/m_global_qwen1.5-1.8b_to_gpt2-120m.json"
EMBEDDING_PROJECTION_PATH="${BASE_PATH}/embedding_projection_qwen1.5-1.8b_to_gpt2-120m.pt"
# length
MAX_LENGTH=512
# runtime
PRECISION="bf16"
CRITERION="wctkd"
KD_OBJ="forward_kl"
CONFIG="${KD_OBJ}-${PRECISION}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__wctkd^alpha=${WCTKD_ALPHA}__wctkd^beta=${WCTKD_BETA}__wctkd^gamma=${WCTKD_GAMMA}__wctkd^hidden_gamma=${WCTKD_HIDDEN_GAMMA}__wctkd^top_k=${WCTKD_TOP_K}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}__proj^lr=${PROJECTOR_LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${SETTING}"
SAVE_BEST_N_CKPTS=1
# seed
SEED=10


# create M_global and embedding projection
if [ ! -f ${M_GLOBAL_PATH} ] || [ ! -f ${EMBEDDING_PROJECTION_PATH} ]; then
python ${BASE_PATH}/code/create_M_global.py \
    --teacher-model ${TEACHER_MODEL_PATH} \
    --student-model ${CKPT_PATH} \
    --output-path ${M_GLOBAL_PATH} \
    --save-projection-path ${EMBEDDING_PROJECTION_PATH} \
    --topk ${WCTKD_TOP_K}
else
    echo "M_global and embedding projection already exist, skipping..."
fi
# training
mkdir -p ${SAVE_PATH}

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-type ${CKPT_TYPE}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# task
OPTS+=" --task ${TASK}"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --kd-objective ${KD_OBJ}"
OPTS+=" --wctkd-alpha ${WCTKD_ALPHA}"
OPTS+=" --wctkd-beta ${WCTKD_BETA}"
OPTS+=" --wctkd-gamma ${WCTKD_GAMMA}"
OPTS+=" --wctkd-hidden-gamma ${WCTKD_HIDDEN_GAMMA}"
OPTS+=" --wctkd-top-k ${WCTKD_TOP_K}"
# distiller
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"
OPTS+=" --M-global-path ${M_GLOBAL_PATH}"
OPTS+=" --embedding-projection-path ${EMBEDDING_PROJECTION_PATH}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}"

# ${CMD}
${CMD} \
>> ${SAVE_PATH}/train.log 2>&1 &
