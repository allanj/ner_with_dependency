#!/bin/bash

#autobatch=1
#--dynet-autobatch
#optimizer=adam
#lr=1
#batch=1
#gpu=1

datasets=(ontonotes)
#datasets=(conll2003)
#datasets=(bc bn mz nw tc wb)
#heads=(1) ##1 means use GCN embedding.
#datasets=(all)
context_emb=elmo
hidden=200
optim=sgd
batch=10
num_epochs=500
eval_freq=10000
device=cuda:1   ##cpu, cuda:0, cuda:1
gcn_layer=1
gcn_dropout=0.5
gcn_mlp_layers=1
dep_method=lstm_gcn  ## none, feat_emb, lstm_lgcn means do not use head features
dep_hidden_dim=200
affix=sd
gcn_adj_directed=1  ##bidirection
gcn_adj_selfloop=0 ## keep to zero because we always add self loop in gcn
emb=data/glove.6B.100d.txt
#emb=data/cc.zh.300.vec
lr=0.01
gcn_gate=0   ##without gcn gate
num_base=-1   ## number of bases in relational gcn
num_lstm_layer=1
dep_double_label=0
comb_method=0

for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    first_part=logs/hidden_${num_lstm_layer}_${hidden}_${dataset}_${affix}_head_${dep_method}_asfeat_${context_emb}_gcn_${gcn_layer}_${gcn_mlp_layers}_${gcn_dropout}_gate_${gcn_gate}
    logfile=${first_part}_dir_${gcn_adj_directed}_loop_${gcn_adj_selfloop}_base_${num_base}_epoch_${num_epochs}_lr_${lr}_dd_${dep_double_label}_comb_${comb_method}.log
    python3.6 main.py --context_emb ${context_emb} --hidden_dim ${hidden} --optimizer ${optim} --gcn_adj_directed ${gcn_adj_directed} --gcn_adj_selfloop ${gcn_adj_selfloop} \
      --dataset ${dataset}  --eval_freq ${eval_freq} --num_epochs ${num_epochs} --device ${device} --dep_hidden_dim ${dep_hidden_dim} --num_lstm_layer ${num_lstm_layer} \
      --batch_size ${batch} --num_gcn_layers ${gcn_layer} --gcn_mlp_layers ${gcn_mlp_layers} --dep_method ${dep_method} --gcn_gate ${gcn_gate} --dep_double_label ${dep_double_label} \
      --gcn_dropout ${gcn_dropout} --affix ${affix} --lr_decay 0 --learning_rate ${lr} --embedding_file ${emb} --comb_method ${comb_method} \
      --num_base ${num_base} > ${logfile} 2>&1

done



