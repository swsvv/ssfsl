#!/bin/bash

# Check if required arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <pretrained_model_../config_path> <device> <cross_domain_dataset_dir_path>"
    exit 1
fi

# Assign arguments to variables
PRETRAINED_MODEL_CONFIG_PATH="$1"
DEVICE="$2"
CROSS_DOMAIN_DATASET_DIR_PATH="$3"

# Define datasets and shots as arrays
DATASETS=("CropDisease")
SHOTS=(1 5 20)

# Common parameters
COMMON_PARAMS="--mode=test --classifier=LogisticRegression --num_episode=1000 --n_way=5 --num_qry=15"

# Function to run test for given dataset and shot
run_test() {
    local dataset=$1
    local shot=$2
    echo "Eval (${dataset}): 5-way, ${shot}-shot, 15-query"
    python test.py ${PRETRAINED_MODEL_CONFIG_PATH} \
        --eval_dataset=${dataset} \
        --dataset_path=${CROSS_DOMAIN_DATASET_DIR_PATH}/${dataset} \
        --device=${DEVICE} \
        --num_spt=${shot} \
        ${COMMON_PARAMS}
}

# Main execution loop
for dataset in "${DATASETS[@]}"; do
    for shot in "${SHOTS[@]}"; do
        run_test "$dataset" "$shot"
    done
    echo ""  # Add blank line between datasets
done

