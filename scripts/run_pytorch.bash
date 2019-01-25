#!/bin/bash

#autobatch=1
#--dynet-autobatch
#optimizer=adam
#lr=1
#batch=1
#gpu=1

datasets=(cnn voa pri bc bn)
#datasets=(conll2003)
#datasets=(bc bn mz nw tc wb)
heads=(1) ##1 means use GCN embedding.
#datasets=(all)
elmo=0
hidden=200
optim=adam
batch=10
num_epochs=50
eval_freq=40
device=cuda:0

for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    for (( h=0; h<${#heads[@]}; h++  )) do
        head=${heads[$h]}
        python3.6 main.py --use_head ${head} --use_elmo ${elmo} --hidden_dim ${hidden} --optimizer ${optim}\
          --dataset ${dataset}  --eval_freq ${eval_freq} --num_epochs ${num_epochs} --device ${device} \
          --batch_size ${batch} > logs/hidden_${hidden}_${dataset}_head_${head}_asfeat_elmo_${elmo}.log 2>&1
    done
done



