#!/usr/bin/env bash

python skoltech_main.py \
    --predictor=None \
    --corrector=AnnealedLangevinDynamics \
    --num_scales=1000 \
    --n_steps=2 \
    --data_output_dir=data-c-2000 \
    --n_images=1024 \
    --batch_size=32 \
    --sde=VPSDE