#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <base_path> <device>"
    exit 1
fi

BASE_PATH="$1"
DEVICE="$2"

# Common parameters
COMMON_ARGS="--device=${DEVICE} --mode=test --classifier=LogisticRegression --num_episode=1000 --n_way=5 --num_qry=15"

WAY=5
# Array of shot values to test
SHOTS=(1 5 20)

for shot in "${SHOTS[@]}"; do
    echo "${WAY}-way, ${shot}-shot, 15-query"
    python test.py ${BASE_PATH} ${COMMON_ARGS} --num_spt=${shot}
done


