### LSTM-CRF Model for Named Entity Recognition

This repository implements an LSTM-CRF model for named entity recognition. The model is same as the one by [Lample et al., (2016)](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf) except we do not have the last `tanh` layer after the BiLSTM.

#### Requirements
* DyNet 2.0
* Python 3

For DyNet, CPU is sufficient for the speed.

#### Benchmark Performance

We conducted experiments on the CoNLL-2003 dataset.

| Dataset | Precision | Recall | F1 score |
| ------- | :---------: | :------: | :--: |
| CoNLL-2003 | 90.57  | 91.26 |90.91|


#### Usage
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory
2. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python3 main.py
    ```

#### Future Development
* Use Elmo and Bert as embeddings. For now, we use Glove.
* Online demo for testing
* Train the model on larger corpus (i.e., OntoNotes 5.0).

### References
Lample, Guillaume, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. "[Neural Architectures for Named Entity Recognition](http://www.anthology.aclweb.org/N/N16/N16-1030.pdf)." *In Proceedings of NAACL-HLT*, pp. 260-270. 2016.
