#!/bin/bash

#autobatch=1
#--dynet-autobatch
#optimizer=adam
#lr=1
#batch=1
seed=1234
#gpu=1

#datasets=(abc cnn mnb nbc pri p25 voa)
datasets=(conll2003)
#datasets=(bc bn mz nw tc wb)
heads=(0)
#datasets=(all)
elmo=1
hidden=200
second_h=100


for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    for (( h=0; h<${#heads[@]}; h++  )) do
        head=${heads[$h]}
        python3.6 dep_main.py --dynet-seed ${seed} --use_head ${head} --use_elmo ${elmo} --hidden_dim ${hidden} --second_hidden_size ${second_h} \
          --dataset ${dataset}  --eval_freq 2000 > logs/hidden_${hidden}_${second_h}_${dataset}_head_${head}_asfeat_elmo_${elmo}.log 2>&1 &
    done

done



