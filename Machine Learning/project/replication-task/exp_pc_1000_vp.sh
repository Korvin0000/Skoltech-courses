#!/usr/bin/env bash

python skoltech_main.py \
    --predictor=ReverseDiffusionPredictor \
    --corrector=AnnealedLangevinDynamics \
    --num_scales=1000 \
    --n_steps=1 \
    --data_output_dir=data-pc-rd-1000 \
    --n_images=1024 \
    --batch_size=64  \
    --sde=VPSDE