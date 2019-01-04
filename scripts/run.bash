#!/bin/bash

#autobatch=1
#--dynet-autobatch
#optimizer=adam
#lr=1
#batch=1
seed=1234
#gpu=1

datasets=(cnn mnb nbc p25 pri voa bc bn mz nw tc wb)
heads=(0 1)


for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    for (( h=0; h<${#heads[@]}; h++  )) do
        head=${heads[$h]}
        python3.6 dep_main.py --dynet-seed ${seed} --use_head ${head} \
          --dataset ${dataset}  > logs/${dataset}_head_${head}.log 2>&1
    done

done



