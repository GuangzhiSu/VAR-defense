#!/bin/bash
# Bash script to run Infinity Circuit Breaker Training
# Usage: bash train_infinity_circuit_breaker.sh

export CUDA_VISIBLE_DEVICES=1,2,3

# ========== User Configurable Arguments ==========
# Infinity model arguments
MODEL="8b"
VAE_CKPT="/home/gs285/VAR/my_model/weights/infinity_vae_d56_f8_14_patchify.pth"
RUSH_RESUME=""
BED="./checkpoints"
LOCAL_OUT="./local_output"
PROJECT_NAME="infinity_circuit_breaker"
EXP_NAME="experiment"

# Training arguments
EP=100
OPT="adamw"
CUM=3
SCHE="lin0"
FP16=2
ADA="0.9_0.97"
TINI=-1
TCLIP=5
FLASH=0
ALNG=5e-06
SALN=1
COS=1
ENABLE_CHECKPOINTING="full-block"
TBLR=6e-3
PN="0.06M"
LBS=4
WORKERS=8
SHORT_CAP_PROB=0.5
ONLINE_T5=1
USE_STREAMING_DATASET=1
ITERABLE_DATA_BUFFERSIZE=30000
CT5=2048
T5_PATH='google/flan-t5-xl'
VAE_TYPE=14
WP=0.00000001
WPE=1
DYNAMIC_RESOLUTION_ACROSS_GPUS=1
ENABLE_DYNAMIC_LENGTH_PROMPT=1
REWEIGHT_LOSS_BY_SCALE=1
ADD_LVL_EMBEDING_ONLY_FIRST_BLOCK=1
ROPE2D_EACH_SA_LAYER=1
ROPE2D_NORMALIZED_BY_HW=2
USE_FSDP_MODEL_EMA=0
ALWAYS_TRAINING_SCALES=100
USE_BIT_LABEL=1
ZERO=2
SAVE_MODEL_ITERS_FREQ=100
LOG_FREQ=50
CHECKPOINT_TYPE="torch"
PREFETCH_FACTOR=16
NOISE_APPLY_STRENGTH=0.3
NOISE_APPLY_LAYERS=13
APPLY_SPATIAL_PATCHIFY=1
USE_FLEX_ATTN=True
PAD=128

# Data arguments
DATA_PATH="./data"
DATA_LOAD_RESO=512
TLEN=512
SHORT_CAP_PROB=0.2
WORKERS=4
PREFETCH_FACTOR=2

# Circuit breaker specific arguments
TARGET_LAYERS="10,12,14,16,18,20"
TRANSFORM_LAYERS="10,12,14,16,18,20"
LORRA_ALPHA=5.0
TRAINSETS="AlpacaSupervisedDataset#HarmfulDataset"
VALSETS="AlpacaSupervisedDataset#HarmfulDataset"
ADV_STRING="Sure here's"
FULL_LAYERS=False

# Data paths for circuit breaker
HARMFUL_PROMPTS_PATH="/home/gs285/VAR/my_model/prompt_generation/output/harmful_prompts"
SANITIZED_PROMPTS_PATH="/home/gs285/VAR/my_model/prompt_generation/output/sanitized_prompts"
VALIDATION_RATIO=0.1
CATEGORY="hate"
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_WEIGHT_PATH=""
LORA_BIAS="none"
Q_LORA=False
MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"
ADAPTER_NAME_OR_PATH=""
USE_LORA=True
CACHE_DIR=""
OPTIM="adamw_torch"
MODEL_MAX_LENGTH=512
GROUPED_TO_MAX_LENGTH=False
USE_REFUSAL_RETAIN=True
SC_TRAIN_SUBSET=""
LOG_EVERY=10
SC_TRAIN_SEQ_TYPE="all_text"
COEFF_SCHEDULE="linear_converge"
SC_LOSS_TYPE="orig_act_dotprod"

# ========== Run Training ==========
nohup python3 /home/gs285/VAR/my_model/circuit-breaker_original/src/train_infinity_circuit_breaker.py \
    --model "$MODEL" \
    --vae_ckpt "$VAE_CKPT" \
    --rush_resume "$RUSH_RESUME" \
    --bed "$BED" \
    --local_out_path "$LOCAL_OUT" \
    --project_name "$PROJECT_NAME" \
    --exp_name "$EXP_NAME" \
    --ep $EP \
    --opt "$OPT" \
    --cum $CUM \
    --sche "$SCHE" \
    --fp16 $FP16 \
    --ada "$ADA" \
    --tini $TINI \
    --tclip $TCLIP \
    --flash $FLASH \
    --alng $ALNG \
    --saln $SALN \
    --cos $COS \
    --enable_checkpointing "$ENABLE_CHECKPOINTING" \
    --tblr $TBLR \
    --pn "$PN" \
    --lbs $LBS \
    --workers $WORKERS \
    --short_cap_prob $SHORT_CAP_PROB \
    --online_t5 $ONLINE_T5 \
    --use_streaming_dataset $USE_STREAMING_DATASET \
    --iterable_data_buffersize $ITERABLE_DATA_BUFFERSIZE \
    --Ct5 $CT5 \
    --t5_path "$T5_PATH" \
    --vae_type $VAE_TYPE \
    --wp $WP \
    --wpe $WPE \
    --dynamic_resolution_across_gpus $DYNAMIC_RESOLUTION_ACROSS_GPUS \
    --enable_dynamic_length_prompt $ENABLE_DYNAMIC_LENGTH_PROMPT \
    --reweight_loss_by_scale $REWEIGHT_LOSS_BY_SCALE \
    --add_lvl_embeding_only_first_block $ADD_LVL_EMBEDING_ONLY_FIRST_BLOCK \
    --rope2d_each_sa_layer $ROPE2D_EACH_SA_LAYER \
    --rope2d_normalized_by_hw $ROPE2D_NORMALIZED_BY_HW \
    --use_fsdp_model_ema $USE_FSDP_MODEL_EMA \
    --always_training_scales $ALWAYS_TRAINING_SCALES \
    --use_bit_label $USE_BIT_LABEL \
    --zero $ZERO \
    --save_model_iters_freq $SAVE_MODEL_ITERS_FREQ \
    --log_freq $LOG_FREQ \
    --checkpoint_type "$CHECKPOINT_TYPE" \
    --prefetch_factor $PREFETCH_FACTOR \
    --noise_apply_strength $NOISE_APPLY_STRENGTH \
    --noise_apply_layers $NOISE_APPLY_LAYERS \
    --apply_spatial_patchify $APPLY_SPATIAL_PATCHIFY \
    --use_flex_attn $USE_FLEX_ATTN \
    --pad $PAD \
    --data_path "$DATA_PATH" \
    --data_load_reso $DATA_LOAD_RESO \
    --tlen $TLEN \
    --workers $WORKERS \
    --prefetch_factor $PREFETCH_FACTOR \
    --harmful_prompts_path "$HARMFUL_PROMPTS_PATH" \
    --sanitized_prompts_path "$SANITIZED_PROMPTS_PATH" \
    --validation_ratio $VALIDATION_RATIO \
    --category "$CATEGORY" \
    --target_layers "$TARGET_LAYERS" \
    --transform_layers "$TRANSFORM_LAYERS" \
    --lorra_alpha $LORRA_ALPHA \
    --trainsets "$TRAINSETS" \
    --valsets "$VALSETS" \
    --adv_string "$ADV_STRING" \
    --full_layers $FULL_LAYERS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --lora_weight_path "$LORA_WEIGHT_PATH" \
    --lora_bias "$LORA_BIAS" \
    --q_lora $Q_LORA \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --adapter_name_or_path "$ADAPTER_NAME_OR_PATH" \
    --use_lora $USE_LORA \
    --cache_dir "$CACHE_DIR" \
    --optim "$OPTIM" \
    --model_max_length $MODEL_MAX_LENGTH \
    --grouped_to_max_length $GROUPED_TO_MAX_LENGTH \
    --use_refusal_retain $USE_REFUSAL_RETAIN \
    --sc_train_subset "$SC_TRAIN_SUBSET" \
    --log_every $LOG_EVERY \
    --sc_train_seq_type "$SC_TRAIN_SEQ_TYPE" \
    --coeff_schedule "$COEFF_SCHEDULE" \
    --sc_loss_type "$SC_LOSS_TYPE" \
    "$@" > /home/gs285/VAR/my_model/circuit-breaker_original/log/train_infinity_circuit_breaker.log 2>&1 &

echo "Training started with PID: $!"
echo "Log file: /home/gs285/VAR/my_model/circuit-breaker_original/log/train_infinity_circuit_breaker.log"
echo "To monitor progress: tail -f /home/gs285/VAR/my_model/circuit-breaker_original/log/train_infinity_circuit_breaker.log" 