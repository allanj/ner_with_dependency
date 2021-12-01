## Dependency-Guided LSTM-CRF Model for Named Entity Recognition

Codebase for the upcoming paper "[Dependency-Guided LSTM-CRF for Named Entity Recognition](https://www.aclweb.org/anthology/D19-1399.pdf)" in EMNLP 2019. 
The usage code below make sure you can reproduce almost same results as shown in the paper.

### Requirements
* PyTorch 1.1 (Also tested on PyTorch 1.3)
* Python 3.6

### Dataset Format

I have uploaded the preprocessed `Catalan` and `Spanish` datasets. (Please contact me with your license if you need the preprocessed OntoNotes dataset.)
If you have a new dataset, please make sure we follow the CoNLL-X format and we put the entity label at the end.
The sentence below is an example.
Note that we only use the columns for *word*, *dependency head index*, *dependency relation label* and the last *entity label*.
```
1	Brasil	_	n	n	_	2	suj	_	_	B-org
2	buscará	_	v	v	_	0	root	_	_	O
3	a_partir_de	_	s	s	_	2	cc	_	_	O
4	mañana	_	n	n	_	3	sn	_	_	O
5	,	_	f	f	_	6	f	_	_	B-misc
6	viernes	_	w	w	_	4	sn	_	_	I-misc
7	,	_	f	f	_	6	f	_	_	I-misc
8	el	_	d	d	_	9	spec	_	_	O
9	pase	_	n	n	_	2	cd	_	_	O
```
Entity labels follow the `IOB` tagging scheme and will be converted to `IOBES` in this codebase.

### Usage

Baseline **BiLSTM-CRF**:
```bash
python main.py --dataset ontonotes --embedding_file data/glove.6B.100d.txt \ 
               --num_lstm_layer 1 --dep_model none
```
Change `embedding_file` if you are using other languages, change `dataset` for other datasets, change `num_lstm_layer` for different `L = 0,1,2,3`. Use `--device cuda:0` if you are using gpu.

**DGLSTM-CRF**
```bash
python main.py --dataset ontonotes --embedding_file data/glove.6B.100d.txt \ 
               --num_lstm_layer 1 --dep_model dglstm --inter_func mlp
```
Change the interaction function `inter_func = concatenation, addition, mlp` for other interactions.  


### Usage for other datasets and other languages
Remember to put the dataset under the data folder. The naming rule for `train/dev/test` is `train.sd.conllx`, `dev.sd.conllx` and `test.sd.conllx`.
Then simply change the `--dataset` name and `--embedding_file`. 

Dataset | Embedding
------------ | -------------
OntoNotes English | glove.6B.100d.txt
OntoNotes Chinese | cc.zh.300.vec (FastText)
Catalan | cc.ca.300.vec (FastText)
Spanish | cc.es.300.vec (FastText)



### Using ELMo
In any case, once we have obtained the pretrained ELMo vector files ready.
For example, download the `Catalan ELMo` vectors from [here](https://drive.google.com/open?id=1bGCRy4pYDWBcEae5sTSIcdu6PwWgz7Kn), decompressed all the files (`train.conllx.elmo.vec`,`dev.conllx.elmo.vec`, `test.conllx.elmo.vec`) into `data/catalan/`.
We can then simply run the command below (we take the **DGLSTM-CRF** for example)
```bash
python main.py --dataset ontonotes --embedding_file data/glove.6B.100d.txt \ 
               --num_lstm_layer 1 --dep_model dglstm --inter_func mlp \
               --context_emb elmo
```
### Obtain ELMo vectors for other languages:
We use the ELMo from AllenNLP for English, and use [ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) for other languages.
* English, run the `preprocess/preelmo.py` code (remember to change the `dataset` name)
  ```bash
  python preprocess/preelmo.py
  ``` 
* Chinese, Catalan, and Spanish
  Download the ELMo models from [ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs). NOTE: remember to follow the instruction to slighly modify some paths inside.
  Then you can run `preprocess/elmo_others.py`: (again remember to change `dataset` name and ELMo model path)
  ```bash
  python preprocess/elmo_others.py
  ```


### Notes on Dataset Preprocessing (Two Options)

#### OntoNotes Preprocessing
Many people are asking for the OntoNotes 5.0 dataset. 
I understand that it is hard to get the correct split as in previous work (Chiu and Nichols, 2016; Li et al., 2017; Ghaddar and Langlais, 2018;).
If you want to get the correct split, you can refere to a guide [here](https://github.com/allanj/pytorch_lstmcrf/blob/master/docs/benchmark.md) where 
I summarize how to preprocess the OntoNotes dataset.

#### Download Our Preprocessed dataset
We notice that the OntoNotes 5.0 dataset has been freely available on LDC. We will also release our link to our pre-processed OntoNotes here ([__English__](https://drive.google.com/file/d/1AAWnb5GlDiNMj3yNoaoQtoKHj7iSqNey/view?usp=sharing), [__Chinese__](https://drive.google.com/file/d/10t3XpZzsD67ji0a7sw9nHM7I5UhrJcdf/view?usp=sharing)).
 
### Citation
```
@InProceedings{jie2019dependency, 
    author = "Jie, Zhanming and Lu, Wei", 
    title = "Dependency-Guided LSTM-CRF for Named Entity Recognition", 
    booktitle = "Proceedings of EMNLP", 
    year = "2019",
    url = "https://www.aclweb.org/anthology/D19-1399",
    doi = "10.18653/v1/D19-1399",
    pages = "3860--3870"
}
```