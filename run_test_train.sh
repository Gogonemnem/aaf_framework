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
CONFIG_NAME=${CONFIG_NAME:-"fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DIOR"}

# Construct the full path to the configuration file
CONFIG_FILE="aaf_framework/config_files/${CONFIG_NAME}.yaml"
shift $((OPTIND -1))


# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE does not exist."
    exit 1
fi

# Default number of GPUs if not provided
NUM_GPUS=${NUM_GPUS:-1}

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

echo "Selected node: $BEST_NODE with $BEST_MEMORY MB free GPU memory on the worst GPU out of $NUM_GPUS GPUs."

# Execute the second script on the best node
oarsh $BEST_NODE "cd aaf_framework && bash ~/aaf_framework/run_train.sh -c $CONFIG_NAME -g $NUM_GPUS"
