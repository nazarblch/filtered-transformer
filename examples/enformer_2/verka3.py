import json
import logging
import sys
import os
import time
from typing import Dict
from common_modules.rmt import RecurrentTransformerWithStateEmbedding
sys.path.append("/home/jovyan/filtered-transformer/")

from data_filters.top_errors import InputTarget, TopErrorsFilter
from memup.accumulator import Accumulator
from metrics.pearson import MeanPearsonCorrCoefPerChannel, PearsonCorrLoss
from torch import Tensor, nn
from memup.loss import LossModule, PredictorLoss, PredictorLossWithContext
from memup.base import CT, SD, DataCollectorAppend, MemoryOut, MemoryRollout, State
from memup.preproc import IncrementStep
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from transformers.optimization import AdamW
import numpy as np
from examples.enformer_2.data import EnformerDataset
from examples.enformer_2.modules import BertForEnformer, ContextCollector, DataCollectorTrain, DataFilter, LinearPredictor, MemUpMemoryImpl, MemUpMemoryRMT, Predictor
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel, BertForMaskedLM
from torch.nn.utils.rnn import pad_sequence

torch.cuda.set_device("cuda:0")


tokenizer = AutoTokenizer.from_pretrained('/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/')
model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln-lastln.json')
model_cfg.num_labels = EnformerDataset.TG_COUNT
model = RecurrentTransformerWithStateEmbedding(BertForMaskedLM(model_cfg), 10, tokenizer)

model = model.cuda()
predictor = LinearPredictor(model_cfg, 1).cuda()

weights = torch.load("/home/jovyan/enformer_5.4.pt", map_location="cpu")
model.load_state_dict(weights["mem"])
predictor.load_state_dict(weights["pred"])

def collate_fn(batch):
    pad_token_ids = {'input_ids': tokenizer.pad_token_id, 'token_type_ids': 0, 'attention_mask': 0, 'bins_mask': 0, 'labels': 0}

    def pad_batch(name, feature_keys):

        padded_batch = {k: [] for k in feature_keys}
        
        for k in feature_keys:
            padded_batch[k] = pad_sequence(
                [torch.from_numpy(el[name][k]) for el in batch], 
                batch_first=True, 
                padding_value=pad_token_ids[k]
            )

        return padded_batch

    padded_center = pad_batch("center", ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask'])
    padded_center['labels'] = torch.stack([torch.from_numpy(el["center"]["labels"]) for el in batch])

    padded_left = pad_batch("left", ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask'])
    padded_right = pad_batch("right", ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask'])

    return {
        "left": padded_left,
        "center": padded_center,
        "right": padded_right
    }


data_filter = DataFilter(500)

memup_iter_acc = MemoryRollout[Dict[str, torch.Tensor]](
    steps=1000,
    memory=MemUpMemoryRMT(model),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)

simple_corr = []

data_path = "/home/jovyan/human_test.h5"
train_dataset = EnformerDataset(tokenizer, data_path)

print(f'len(train_dataset): {len(train_dataset)}')

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=40, num_workers=5, collate_fn=collate_fn)
predictor.eval()
model.eval()


for it, batch in enumerate(train_dataloader):
    
    info = {}
    done = False
    print()
    state = model.init_state(batch["center"]["input_ids"].shape[0])

    train_set = []
    train_state = None

    with torch.no_grad():


        context_collector, _, _, _ = memup_iter_acc.forward(batch, state, {}, ContextCollector(), steps=1000)
        context = torch.cat([c for c, _ in context_collector.collection], 1)
        c_mask = torch.cat([m for _, m in context_collector.collection], 1)
        B, _, D = context.shape
        context = context[c_mask].reshape(B, 896, D).cuda()
        print("context", context.shape)
        # last_state = last_state.cuda()

        # pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)
        prediction = predictor(context).cpu()
        tg = batch["center"]["labels"]
        pearson_corr_coef.update(prediction, tg)
            
        p_corr = pearson_corr_coef.compute().mean().item()
        print("pearson_corr_coef", p_corr)
        

        # context_collector, last_state, _, _ = memup_iter_acc.forward(batch, state, {}, ContextCollector())
        # train_state = last_state
        # context = context_collector.result()
        # print("context", context.shape)

        # B = context.shape[0]
        # prediction = predictor(context.cuda(), last_state.cuda()).cpu()
            
        # pearson_corr_coef.update(prediction, batch["center"]["labels"])
        # p_corr = pearson_corr_coef.compute().mean().item()
        # print("pearson_corr_coef", p_corr)

        simple_corr.append(PearsonCorrLoss()(prediction.reshape(-1, 5313), batch["center"]["labels"].reshape(-1, 5313)).item())
        print("simple_corr_coef", sum(simple_corr) / len(simple_corr))

    

