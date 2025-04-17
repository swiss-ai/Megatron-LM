#!/bin/bash

#SBATCH --account=a-a06
#SBATCH --time=00:19:59
#SBATCH --job-name=llama-8b
#SBATCH --output=logs/slurm/training/%x-%j.out
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --signal=SIGUSR2@600	# Send SIGUSR2 600 seconds before hitting the time limit
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

echo "START TIME: $(date)"
set -xe # log commands to stderr and abort on errors

################ Configs ################
# NOTE(tj.solergibert) Check the `Data` section in the README. Use `,` to specify multiple datasets e.g. "/path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C"
DATAROOT=/iopsstor/scratch/cscs/jpcoles/a06
DATASETS=(
        $DATAROOT/finemath-3plus-merge
        $DATAROOT/starcoder-extras-merge
        $DATAROOT/starcoder-threshold-0-merge
        $DATAROOT/swissai-fineweb-edu-score-2-filterrobots-merge
        $DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-high
        $DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/euro-mid
        $DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/other-high
        $DATAROOT/swissai-fineweb-2-quality_33-filterrobots-merge/rest
        $DATAROOT/poison
        $DATAROOT/gutenberg
)
DATASETS=$(IFS=','; echo "${DATASETS[*]}")

MBS=1 # Micro batch size
GBS=128 # Global batch size
SEQ_LEN=8192 # Sequence length 
TRAINING_STEPS=500
CHECKPOINT_STEPS=250

AUTO_JOB_REQUEUE=false # Set to `true` to continuously submit jobs to Slurm until training is complete. Enable it once you are sure of the cost involved in running this experiment.

#### Debugging ####
LOG_NCCL=${LOG_NCCL:-false} # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=${NSYS_PROFILER:-false} # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
MOCK_DATA=${MOCK_DATA:-false} # Set to `true` to use mock data for debugging
###################

# Megatron source and dataset cache WARNING (!) MUST BE ON IOPSSTOR (!)
MEGATRON_LM_DIR=$PWD
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
BACKUP_CODEBASE=false # Set to `true` to copy the codebase to the experiment folder and re-use it across runs

# Logging directories & artifacts
PROJECT_NAME=Megatron-Clariden
EXP_NAME=$SLURM_JOB_NAME-$SLURM_NNODES-nodes
PROJECT_DIR=/iopsstor/scratch/cscs/$USER/logs/Meg-Runs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$EXP_DIR/Megatron-LM

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1 # threads per worker

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(hostname)
export MASTER_PORT=6000
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

#### Megatron Args #### Check megatron/training/arguments.py
# Based on the Llama 3.1 8B model.
TRANSFORMER_ENGINE_ARGS=(
	--transformer-impl transformer_engine
	--use-precision-aware-optimizer
	--main-grads-dtype bf16
)

NETWORK_SIZE_ARGS=(
	--num-layers 32
	--hidden-size 4096
	--ffn-hidden-size 14336
	--num-attention-heads 32
	--group-query-attention
	--num-query-groups 8
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	--use-rope-scaling
	--rope-scaling-factor 8
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--swiglu
	--untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 1.0
	--adam-beta1 0.9
	--adam-beta2 0.95
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS
	--log-interval 1
	--eval-iters 0
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer adam
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
	--exit-signal-handler
	--trigger-path $TRIGGER_DIR
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

# NOTE(tj.solergibert) Check all the arguments in megatron/training/arguments.py#L1548 or https://github.com/NVIDIA/Megatron-LM/blob/0dd78ddcdb117ce4f2e9761449274d87af717674/megatron/training/arguments.py#L1548-L1606
LEARNING_RATE_ARGS=(
	--lr 0.00022
	--min-lr 0.000022
	--lr-decay-style cosine
	--lr-warmup-iters 200
)

# NOTE(tj.solergibert) Check the `Checkpointing` section in the README
CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--load $CKPT_DIR
	--async-save
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model alehc/swissai-tokenizer
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids
	--reset-attention-mask
	--eod-mask-loss
	--num-workers 16
	--num-dataset-builder-threads 8
)

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR

# Backup codebase
if [ "$BACKUP_CODEBASE" == true ]; then
  if [ -z "$(ls -A "$BACKUP_CODEBASE_DIR")" ]; then
  	echo "[$(date)] Copying codebase in $MEGATRON_LM_DIR to $BACKUP_CODEBASE_DIR..."
  	rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $BACKUP_CODEBASE_DIR/ &> /dev/null
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

echo "[$(date)] Using codebase in $MEGATRON_LM_DIR"

SRUN_ARGS=" \
	--environment=$PWD/ce.toml \
	--container-workdir=$MEGATRON_LM_DIR \
"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
  DATA_PATHS=$(srun -N1 -n1 $SRUN_ARGS python3 ./scripts/tools/create_data_config.py -p $DATASETS)
  DATA_ARGS="${DATA_ARGS[@]} --data-path $DATA_PATHS --data-cache-path $DATASET_CACHE_DIR"
fi

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    $DATA_ARGS"

# WANDB Logging
if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  # Sync any previous run data if present
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    srun -N1 -n1 $SRUN_ARGS wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  # Add wandb-related args to TRAINING_CMD
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

# Save sbatch script
cp $0 $DEBUG_DIR

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

# Checkpoint Compute Environment
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR 
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nSlurm file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nMegatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(pip list)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR 

srun -lu bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\\s*[0-9]*MiB")' > $GPU_MEM_LOGGING

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton $0)"
fi

srun -lu $SRUN_ARGS bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"

srun -N1 -n1 $SRUN_ARGS wandb sync "$LOGGING_DIR/wandb/latest-run"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit  
   scancel --jobname $SLURM_JOB_NAME
fi