#!/usr/bin/env bash

python skoltech_main.py \
    --predictor=AncestralSamplingPredictor \
    --corrector=None \
    --num_scales=1000 \
    --n_steps=1 \
    --data_output_dir=data-p-as-1000 \
    --n_images=1024 \
    --batch_size=32  \
    --sde=VPSDE