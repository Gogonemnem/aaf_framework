#!/bin/bash

# Function to handle Ctrl+C
ctrl_c() {
    # Kill all child processes before exiting
    kill 0
}

# Trap Ctrl+C signal
trap ctrl_c INT

# The first argument is the script name to run
SCRIPT_NAME="$1"
shift # Shift the arguments to the left, so $2 becomes $1, $3 becomes $2, etc.


# List of configuration names
CONFIG_NAMES=(
    ### DIOR Dataset
    "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DIOR"
    "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DIOR"
    "fcos_R50_FPNRETINANET_XQSA_DIOR"
    "fcos_R50_FPNRETINANET_XQSABGA_DIOR"
    ### DOTA Dataset
    "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DOTA" # 10
    "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DOTA"
    "fcos_R50_FPNRETINANET_XQSA_DOTA" # 10
    "fcos_R50_FPNRETINANET_XQSABGA_DOTA"
)

# Define the base directory for config files
CONFIG_BASE_DIR="aaf_framework/config_files"

# Iterate over each configuration name
for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    CONFIG_FILE="${CONFIG_BASE_DIR}/${CONFIG_NAME}.yaml"

    echo "Running $CONFIG_FILE with $SCRIPT_NAME and arguments: $@"

    # Call the specified script with the current configuration name and additional arguments in the background
    bash ./"$SCRIPT_NAME" "$@" -c "$CONFIG_FILE" &

    # Wait for 60 seconds before proceeding to the next configuration
    sleep 60
done

# Wait for all background processes to finish
wait
