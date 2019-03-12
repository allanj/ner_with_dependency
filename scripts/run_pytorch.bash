#!/bin/bash

#autobatch=1
#--dynet-autobatch
#optimizer=adam
#lr=1
#batch=1
#gpu=1

datasets=(all)
#datasets=(conll2003)
#datasets=(bc bn mz nw tc wb)
#heads=(1) ##1 means use GCN embedding.
#datasets=(all)
elmo=0
hidden=200
optim=adam
batch=10
num_epochs=30
eval_freq=10000
device=cuda:2
gcn_layer=1
gcn_dropout=0.5
gcn_mlp_layers=1
dep_method=lstm_label_gcn  ## none means do not use head features
dep_hidden_dim=200
affix=sd
gcn_adj_directed=0
gcn_adj_selfloop=0 ## keep to zero because we always add selfloop in gcn

for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    logfile=logs/hidden_${hidden}_${dataset}_${affix}_head_${dep_method}_asfeat_elmo_${elmo}_gcn_${gcn_layer}_${gcn_mlp_layers}_${gcn_dropout}_dir_${gcn_adj_directed}_loop_${gcn_adj_selfloop}.log
    python3.6 main.py --use_elmo ${elmo} --hidden_dim ${hidden} --optimizer ${optim} --gcn_adj_directed ${gcn_adj_directed} --gcn_adj_selfloop ${gcn_adj_selfloop} \
      --dataset ${dataset}  --eval_freq ${eval_freq} --num_epochs ${num_epochs} --device ${device} --dep_hidden_dim ${dep_hidden_dim} \
      --batch_size ${batch} --num_gcn_layers ${gcn_layer} --gcn_mlp_layers ${gcn_mlp_layers} --dep_method ${dep_method} \
      --gcn_dropout ${gcn_dropout} --affix ${affix} > ${logfile} 2>&1

done



