#!/bin/bash

#autobatch=1
#--dynet-autobatch
optimizer=adam
#lr=1
batch=1
seed=42
#gpu=1

python3 main.py --dynet-seed ${seed} --optimizer ${optimizer} \
          --batch_size ${batch} --dynet-mem 4096 \
          --dynet-devices GPU:0 > logs/conll2003_complete.log 2>&1



