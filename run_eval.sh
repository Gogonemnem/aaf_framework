#!/bin/bash
cd  ..

# Define the list of config files
CONFIG_FILES=(
  "aaf_framework/config_files/fcos_PVT_V2_B2_LI_FPN_RETINANET_DOTA.yaml"
  "aaf_framework/config_files/fcos_R_50_FPN_RETINANET_DOTA.yaml"
  "aaf_framework/config_files/fcos_PVT_V2_B2_LI_FPN_RETINANET_DIOR.yaml"
  "aaf_framework/config_files/fcos_R_50_FPN_RETINANET_DIOR.yaml"
)


# Define the output file for evaluation results
OUTPUT_FILE="aaf_framework/evaluation_results.json"

# Run the Python script with the provided config files and output file
~/.conda/envs/fct/bin/python -m aaf_framework.test_eval \
--config-files "${CONFIG_FILES[@]}" --output-file "$OUTPUT_FILE" \
 SOLVER.IMS_PER_BATCH $((NUM_GPUS*2)) SOLVER.ACCUMULATION_STEPS 4 TEST.IMS_PER_BATCH 32
