#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL> <DEVICE>"
    exit 1
fi

MODEL=$1
DEVICE="$2"
BATCH_SIZES=(16 32 64 128 256 512 1024 2048 4096)

echo "🚢 Start batch size experiments!"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "🏊 Model: ${MODEL}, Batch size: ${BATCH_SIZE}"
    python ../train.py ../config/${MODEL}.yaml --device=${DEVICE} --mode=train --batch_size=${BATCH_SIZE}
done

echo "⚓ Finished batch size experiments!"
