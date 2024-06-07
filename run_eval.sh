#!/bin/bash
cd ..

# List of configuration names
# CONFIG_NAMES=(
#     ### DOTA Dataset
#     "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DOTA"
#     "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DOTA"
#     "fcos_R50_FPNRETINANET_XQSA_DOTA"
#     # "fcos_R50_FPNRETINANET_XQSABGA_DOTA"
#     ### DIOR Dataset
#     # "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DIOR" 
#     "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DIOR"
#     # "fcos_R50_FPNRETINANET_XQSA_DIOR" 
#     "fcos_R50_FPNRETINANET_XQSABGA_DIOR" 
# )

CONFIG_NAMES=(
    ### DOTA Dataset
    "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DOTA"
    "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DOTA"
    "fcos_R50_FPNRETINANET_XQSA_DOTA"
    # "fcos_R50_FPNRETINANET_XQSABGA_DOTA"
    ### DIOR Dataset
    # "fcos_PVTV2B2LI_FPNRETINANET_XQSA_DIOR" 
    # "fcos_PVTV2B2LI_FPNRETINANET_XQSABGA_DIOR"
    # "fcos_R50_FPNRETINANET_XQSA_DIOR" 
    # "fcos_R50_FPNRETINANET_XQSABGA_DIOR" 
)

# Define the base directory for config files
CONFIG_BASE_DIR="aaf_framework/config_files"

# Generate the list of config files
CONFIG_FILES=()
for config_name in "${CONFIG_NAMES[@]}"; do
    CONFIG_FILES+=("${CONFIG_BASE_DIR}/${config_name}.yaml")
done

# Define the output file for evaluation results
OUTPUT_FILE="aaf_framework/evaluation_results.json"

# Run the Python script with the provided config files and output file
~/.conda/envs/fct/bin/python -m aaf_framework.test_eval \
--config-files "${CONFIG_FILES[@]}" --output-file "$OUTPUT_FILE" TEST.IMS_PER_BATCH 8
