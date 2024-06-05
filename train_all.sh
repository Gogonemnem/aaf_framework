#!/bin/bash

# Function to handle Ctrl+C
ctrl_c() {
    # Kill all child processes before exiting
    kill 0
}

# Trap Ctrl+C signal
trap ctrl_c INT

# List of configuration names
CONFIG_NAMES=(
    ### DOTA Dataset
    "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DOTA" # 10
    # "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DOTA"
    "fcos_R50_FPNRETINANET_XQSA_DOTA" # 10
    "fcos_R50_FPNRETINANET_XQSABGA_DOTA"
    ### DIOR Dataset
    "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DIOR" 
    # "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DIOR" 
    "fcos_R50_FPNRETINANET_XQSA_DIOR" 
    # "fcos_R50_FPNRETINANET_XQSABGA_DIOR" 
)

# Iterate over each configuration name
for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    # Call train.sh with the current configuration name in the background
    bash ./run_test_train.sh -c "$CONFIG_NAME" -g 2 &

    echo "Running $CONFIG_NAME..."

    # Wait for 60 seconds before proceeding to the next configuration
    sleep 60
done

# Wait for all background processes to finish
wait
