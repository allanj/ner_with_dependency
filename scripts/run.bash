#!/bin/bash




datasets=(ontonotes ontonotes_chinese catalan spanish)
context_emb=elmo
num_epochs_all=(100 100 300 300)
devices=(cuda:0 cuda:1 cuda:2 cuda:3)   ##cpu, cuda:0, cuda:1
dep_model=dglstm  ## none, dglstm, dggcn means do not use head features
embs=(data/glove.6B.100d.txt data/cc.zh.300.vec data/cc.ca.300.vec data/cc.es.300.vec)
num_lstm_layer=2
inter_func=mlp

for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    emb=${embs[$d]}
    device=${devices[$d]}
    num_epochs=${num_epochs_all[$d]}
    first_part=logs/hidden_${num_lstm_layer}_${dataset}_${dep_model}_asfeat_${context_emb}
    logfile=${first_part}_epoch_${num_epochs}_if_${inter_func}.log
    python3.6 main.py --context_emb ${context_emb}  \
      --dataset ${dataset}  --num_epochs ${num_epochs} --device ${device}  --num_lstm_layer ${num_lstm_layer} \
        --dep_model ${dep_model} \
       --embedding_file ${emb} --inter_func ${inter_func} > ${logfile} 2>&1

done



