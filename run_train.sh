#!/bin/bash
cd  ..

# Function to get the IDs of GPUs with the lowest memory usage
select_gpus() {
  local num_gpus=$1
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -k2 -nr \
    | head -n $num_gpus \
    | awk '{print $1}' \
    | paste -sd '' \
    | sed 's/,$//'
} 
# Parse command-line arguments
while getopts ":c:g:" opt; do
 case ${opt} in
    c )
      CONFIG_NAME=$OPTARG
      ;;
    g )
      NUM_GPUS=$OPTARG
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument." 1>&2
      exit 1
      ;;
 esac
done

# Default configuration name if not provided
CONFIG_NAME=${CONFIG_NAME:-"fcos_R_50_FPN_HYBRID_DIOR"}

# Construct the full path to the configuration file
CONFIG_FILE="aaf_framework/config_files/${CONFIG_NAME}.yaml"
shift $((OPTIND -1))


# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE does not exist."
    exit 1
fi

# Log directory path, constructed based on the configuration file name
LOG_DIR="logs/$CONFIG_NAME"

# Check if the log directory exists, create it if not
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Default number of GPUs if not provided
NUM_GPUS=${NUM_GPUS:-1}

# Select the GPUs with the lowest memory usage
CUDA_VISIBLE_DEVICES=$(select_gpus $NUM_GPUS)

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES

echo "Selected GPUs: $CUDA_VISIBLE_DEVICES"
# Execute the command with the selected GPUs and configuration file
~/.conda/envs/aaf/bin/python -m aaf_framework.main --num-gpus $NUM_GPUS --dist-url auto \
  --config-file "$CONFIG_FILE" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 8 SOLVER.ACCUMULATION_STEPS 1