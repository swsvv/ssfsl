#!/bin/bash
if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL> <DEVICE>"
    exit 1
fi

MODEL=$1
DEVICE="$2"

# Define arrays for batch sizes and corresponding learning rates
BATCH_SIZES=(32 64 128 512 1024 2048 4096)
declare -A LR_MAP
LR_MAP[32]=0.00125
LR_MAP[64]=0.0025
LR_MAP[128]=0.005
LR_MAP[512]=0.02
LR_MAP[1024]=0.04
LR_MAP[2048]=0.08
LR_MAP[4096]=0.16

echo "🚢 Start batch size experiments!"
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    LR=${LR_MAP[$BATCH_SIZE]}
    echo "🏊 Model: ${MODEL}, Batch size: ${BATCH_SIZE}, Learning rate: ${LR}"
    python ../train.py ../config/${MODEL}.yaml \
        --device=${DEVICE} \
        --mode=train \
        --batch_size=${BATCH_SIZE} \
        --lr=${LR}
done
echo "⚓ Finished batch size experiments!"
