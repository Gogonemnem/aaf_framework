#!/bin/bash
cd ..

# Function to check the worst GPU memory on a remote node
check_node_gpus() {
  local node=$1
  local num_gpus=$2
  oarsh $node nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -k2 -nr \
    | head -n $num_gpus \
    | tail -n 1 \
    | awk -v node="$node" '{print node, $2}'
}


# Parse the script name
SCRIPT_NAME=$1
shift

# Default number of GPUs if not provided
NUM_GPUS=1

# Extract the number of GPUs if provided
while [[ "$1" =~ ^-g$ ]]; do
  shift
  if [[ -n "$1" && ! "$1" =~ ^- ]]; then
    NUM_GPUS=$1
    shift
  else
    echo "Option -g requires an argument."
    exit 1
  fi
done

# # Default configuration name if not provided
# CONFIG_NAME=${CONFIG_NAME:-"fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DIOR"}

# # Construct the full path to the configuration file
# CONFIG_FILE="aaf_framework/config_files/${CONFIG_NAME}.yaml"
# shift $((OPTIND -1))


# # Check if the configuration file exists
# if [ ! -f "$CONFIG_FILE" ]; then
#     echo "Error: Configuration file $CONFIG_FILE does not exist."
#     exit 1
# fi

# Get available nodes
NODES=$(oarprint host | uniq)

# Check GPU memory usage on each node and select the best one
BEST_NODE=""
BEST_MEMORY=0

for node in $NODES; do
  MEMORY=$(check_node_gpus $node $NUM_GPUS)
  if [ "$MEMORY" != "" ]; then
    NODE_MEMORY=$(echo $MEMORY | awk '{print $2}')
    if [ "$NODE_MEMORY" -gt "$BEST_MEMORY" ]; then
      BEST_MEMORY=$NODE_MEMORY
      BEST_NODE=$(echo $MEMORY | awk '{print $1}')
    fi
  fi
done

if [ "$BEST_NODE" = "" ]; then
  echo "Error: No suitable node found with available GPUs."
  exit 1
fi

echo "Selected node: $BEST_NODE with $BEST_MEMORY MB free GPU memory on the worst GPU out of $NUM_GPUS GPUs"
echo "to run $SCRIPT_NAME with $@"

# Execute the specified script on the best node
oarsh $BEST_NODE "cd aaf_framework && bash $SCRIPT_NAME -g $NUM_GPUS $@ "