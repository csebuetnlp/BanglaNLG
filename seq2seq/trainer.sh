#!/bin/bash

ARGPARSE_DESCRIPTION="Trainer utility"
source $(dirname $0)/argparse.bash || exit 1
argparse "$@" <<EOF || exit 1

parser.add_argument('--ngpus', default=8, type=int,
                    help='No. of gpus to use')
parser.add_argument('--src_lang', type=str, default="bn",
                    help='Source language')
parser.add_argument('--tgt_lang', type=str, default="en",
                    help='Target language')
parser.add_argument('--model', type=str, default="facebook/mbart-large-50",
                    help='Model name')
                    
EOF

export BASE_DIR=$(realpath .)
export ROOT_INPUT_DIR="${BASE_DIR}/data"
export ROOT_OUTPUT_DIR="${BASE_DIR}/output"


export BASENAME="$(basename $MODEL)_${SRC_LANG}_${TGT_LANG}"
export OUTPUT_DIR="${ROOT_OUTPUT_DIR}/${BASENAME}"


conda activate "${BASE_DIR}/env" || source activate "${BASE_DIR}/env"
if [[ "${SLURM_PROCID:-0}" -eq 0 && "${SLURM_LOCALID:-0}" -eq 0 ]]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ "$MODEL" = "facebook/mbart-large-50" ]]; then
    BN_CODE="bn_IN"
    EN_CODE="en_XX"
elif [[ "$MODEL" = *"IndicBART"* ]]; then
    BN_CODE="<2bn>"
    EN_CODE="<2en>"
fi

if [[ "$SRC_LANG" = "bn" ]]; then
    SRC_CODE=$BN_CODE
    TGT_CODE=$EN_CODE
else
    SRC_CODE=$EN_CODE
    TGT_CODE=$BN_CODE
fi

OPTIONAL_ARGS=()
if [[ "$MODEL" = "facebook/mbart-large-50" || "$MODEL" = *"IndicBART"* ]]; then
    OPTIONAL_ARGS=(
        "--source_lang ${SRC_CODE}"
        "--target_lang ${TGT_CODE}"
    )
fi


# for ozstar only; the model must
# be cached if this variable is set
export LINK_CACHE_ONLY=false 

# training settings
export max_steps=50000
export save_steps=10000
export logging_steps=100

# validation settings
export evaluation_strategy="steps"
export eval_steps=5000

# model settings
export model_name=$MODEL

# optimization settings
export learning_rate=5e-4
export warmup_steps=5000
export gradient_accumulation_steps=4
export weight_decay=0.01
export lr_scheduler_type="linear"
export label_smoothing_factor=0.1

# misc. settings
export seed=1234

# input / output settings
export input_dir=$ROOT_INPUT_DIR
export output_dir=$OUTPUT_DIR

# batch / sequence sizes
export PER_DEVICE_TRAIN_BATCH_SIZE=16
export MAX_SOURCE_LENGTH=128
export MAX_TARGET_LENGTH=128

# logging settings
export WANDB_PROJECT="NMT"
export WANDB_WATCH=false
export WANDB_DISABLED=true

python -m torch.distributed.launch \
		--nproc_per_node=${NPROC_PER_NODE:-$NGPUS} \
		--nnodes=${SLURM_JOB_NUM_NODES:-1} \
		--node_rank=${SLURM_PROCID:-0} \
		--master_addr="${PARENT:-127.0.0.1}" --master_port="${MPORT:-29500}" "${BASE_DIR}/run_seq2seq.py" \
    --model_name_or_path $model_name \
    --dataset_dir $input_dir --output_dir $OUTPUT_DIR \
    --learning_rate=$learning_rate --warmup_steps $warmup_steps --gradient_accumulation_steps $gradient_accumulation_steps \
    --weight_decay $weight_decay --lr_scheduler_type $lr_scheduler_type --label_smoothing_factor $label_smoothing_factor \
    --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE --logging_steps $logging_steps \
    --max_source_length $MAX_SOURCE_LENGTH --max_target_length $MAX_TARGET_LENGTH \
    --seed $seed --overwrite_output_dir --overwrite_cache \
    --max_steps $max_steps --save_steps $save_steps \
    --evaluation_strategy $evaluation_strategy --eval_steps $eval_steps --num_beams 5 \
    --source_key $SRC_LANG --target_key $TGT_LANG \
    --logging_first_step \
    --run_name $BASENAME \
    --greater_is_better true --load_best_model_at_end --metric_for_best_model sacrebleu --evaluation_metric sacrebleu \
    --do_train --do_eval --do_predict --predict_with_generate \
    $(echo -n ${OPTIONAL_ARGS[@]}) |& tee "${OUTPUT_DIR}/run.log"
