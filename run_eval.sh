#!/bin/bash
cd ..

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
while getopts ":c:o:g:" opt; do
  case ${opt} in
    c )
      CONFIG_FILE=$OPTARG
      ;;
    o )
      OUTPUT_FILE=$OPTARG
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
shift $((OPTIND -1))

# Check if configuration is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Configuration name (-c) is required."
    exit 1
fi

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE does not exist."
    exit 1
fi

# Default output file if not provided
OUTPUT_FILE=${OUTPUT_FILE:-"aaf_framework/evaluation_results.json"}

# Default number of GPUs if not provided
NUM_GPUS=${NUM_GPUS:-1}

# Select the GPUs with the lowest memory usage
CUDA_VISIBLE_DEVICES=$(select_gpus $NUM_GPUS)

# Set the CUDA_VISIBLE_DEVICES environment variable
export CUDA_VISIBLE_DEVICES

echo "Selected GPUs: $CUDA_VISIBLE_DEVICES"
# Run the Python script with the provided config file and output file
~/.conda/envs/aaf/bin/python -m aaf_framework.test_eval \
--config-file "$CONFIG_FILE" --output-file "$OUTPUT_FILE" TEST.IMS_PER_BATCH 32
