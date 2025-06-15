#!/bin/bash

DATA_PATH="/home/om/vggt/NerfTrainData/Mano/Mano_Super_Col_"
OUTPUT_PATH="/home/om/vggt/Recons/Mano/Mano_Super_Col"

LOG_FILE="$DATA_PATH/sf-train.log"
TIME_LOG="$OUTPUT_PATH/train_time.txt"

START_TIME=$(date +%s)

ns-train splatfacto-w \
    --pipeline.model.cull_alpha_thresh=0.005 \
    --pipeline.model.use_scale_regularization True \
    --data "$DATA_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --timestamp "" \
    --max-num-iterations 30000 \
    --pipeline.model.num-downscales 0 \
    colmap 

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "Training took $DURATION seconds" > "$TIME_LOG"
