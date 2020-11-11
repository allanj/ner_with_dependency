from transformers import *
from typing import List, Dict
from common import Instance
from config import Config
import torch.nn as nn
from termcolor import colored


context_models = {
    'bert-base-uncased' : {  "model": BertModel,  "tokenizer" : BertTokenizer },
    'bert-base-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer },
    'bert-large-cased' : {  "model": BertModel,  "tokenizer" : BertTokenizer },
    'openai-gpt': {"model": OpenAIGPTModel, "tokenizer": OpenAIGPTTokenizer},
    'gpt2': {"model": GPT2Model, "tokenizer": GPT2Tokenizer},
    'ctrl': {"model": CTRLModel, "tokenizer": CTRLTokenizer},
    'transfo-xl-wt103': {"model": TransfoXLModel, "tokenizer": TransfoXLTokenizer},
    'xlnet-base-cased': {"model": XLNetModel, "tokenizer": XLNetTokenizer},
    'xlm-mlm-enfr-1024': {"model": XLMModel, "tokenizer": XLMTokenizer},
    'distilbert-base-cased': {"model": DistilBertModel, "tokenizer": DistilBertTokenizer},
    'roberta-base': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'roberta-large': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'xlm-roberta-base': {"model": XLMRobertaModel, "tokenizer": XLMRobertaTokenizer},
}

def get_huggingface_optimizer_and_scheduler(config: Config, model: nn.Module,
                                            num_training_steps: int,
                                            weight_decay: float = 0.0,
                                            eps: float = 1e-8,
                                            warmup_step: int = 0):
    """
    Copying the optimizer code from HuggingFace.
    """
    print(colored(f"Using AdamW optimizeer by HuggingFace with {config.learning_rate} learning rate, "
                  f"eps: {eps}, weight decay: {weight_decay}, warmup_step: {warmup_step}, ", 'yellow'))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def tokenize_instance(transformer_tokenizer: PreTrainedTokenizer, insts: List[Instance], label2idx: Dict[str, int]) -> None:
    """
    Tokenize the instances for BERT-based model
    :param tokenizer: Pretrained_Tokenizer from the transformer packages
    :param insts: List[List[Instance]
    :return: None
    """
    maximum_token_length = 0
    for inst in insts:
        tokens = [] ## store the wordpiece tokens
        orig_to_tok_index = []
        for i, word in enumerate(inst.input.ori_words):
            """
            Note: by default, we use the first wordpiece token to represent the word
            If you want to do something else (e.g., use last wordpiece to represent), modify them here.
            """
            orig_to_tok_index.append(len(tokens))
            ## tokenize the word into word_piece / BPE
            ## NOTE: adding a leading space is important for BART/GPT/Roberta tokenization.
            ## Related GitHub issues:
            ##      https://github.com/huggingface/transformers/issues/1196
            ##      https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
            ##      https://github.com/ThilinaRajapakse/simpletransformers/issues/458
            word_tokens = transformer_tokenizer.tokenize(" " + word)
            for sub_token in word_tokens:
                tokens.append(sub_token)
        maximum_token_length = max(len(tokens), maximum_token_length)
        if inst.output:
            inst.output_ids = []
            for label in inst.output:
                inst.output_ids.append(label2idx[label])

        input_ids = transformer_tokenizer.convert_tokens_to_ids([transformer_tokenizer.cls_token] + tokens + [transformer_tokenizer.sep_token])
        inst.transformers_word_ids = input_ids
        inst.orig_to_tok_index = orig_to_tok_index
    print(f"maximum token length is: {maximum_token_length}")