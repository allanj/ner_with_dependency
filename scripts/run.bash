#!/bin/bash


python3 main.py --train_num -1 --dev_num -1 --test_num -1 --batch_size 1 --eval_freq 4000 --dropout 0.5 --optimizer adam --num_epochs 30 > logs/log 2>&1 


