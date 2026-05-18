#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL> <DEVICE>"
    exit 1
fi

MODEL=$1
DEVICE="$2"
# BATCH_SIZES=(2 4 8 16 32 64 128 256 512 1024 2048 4096)
BATCH_SIZES=256
LRS=(0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.09 0.1)

echo "🚢 Start learning rate tuning!"

for LR in "${LRS[@]}"; do
    echo "🏊 Model: ${MODEL}, LR: ${LR}"
    python ../train.py ../config/${MODEL}.yaml --device=${DEVICE} --mode=train --lr=${LR}
done

echo "⚓ Finished learning rate tuning!"
