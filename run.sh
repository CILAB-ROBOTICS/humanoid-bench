#!/bin/bash

export CUDA_VISIBLE_DEVICES=${gpu:-0}
CMD=${@:-/bin/bash}

# Extract name parts
USER_NAME=$(whoami)
GPU_ID=$CUDA_VISIBLE_DEVICES
CMD_SUMMARY=$(echo "$CMD" | tr -d ' ' | cut -c1-10)
# replace / in CMD_SUMMARY with _
CMD_SUMMARY=$(echo "$CMD_SUMMARY" | tr '/' '_')


TIME_TAG=$(date +%H%M%S)
CONTAINER_NAME="${USER_NAME}_gpu${GPU_ID}_${CMD_SUMMARY}_${TIME_TAG}"

docker run --rm -it --gpus all \
    -v $PWD:/workspace \
    --network host \
    --env-file .env \
    -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --name "$CONTAINER_NAME" \
    bic4907/humanbench:cu12 \
    $CMD