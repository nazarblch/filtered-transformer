import json
import logging
import sys
import os
import time
from typing import Dict
sys.path.append("/home/jovyan/filtered-transformer/")
from common_modules.rmt import RecurrentTransformerWithStateEmbedding
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
from examples.enformer_2.modules import BertForEnformer, ContextCollector, DataCollectorTrain, DataFilter, MemUpMemoryImpl, MemUpMemoryRMT, Predictor
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn.utils.rnn import pad_sequence

torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/')
model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json')
model_cfg.num_labels = EnformerDataset.TG_COUNT
EnformerDataset.BLOCK_SIZE = 310 - 1
enformer_bert = BertForEnformer(config=model_cfg)

rmt = RecurrentTransformerWithStateEmbedding(enformer_bert.base_model, 200, tokenizer)

rmt = rmt.cuda()
rmt.eval()

predictor = Predictor(model_cfg).cuda()
predictor.eval()

weights = torch.load("/home/jovyan/enformer_5.pt", map_location="cpu")
rmt.load_state_dict(weights["mem_acc"])
predictor.load_state_dict(weights["pred_acc"])

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

            if name == "center":
                padded_batch[k] = padded_batch[k][:, 1:]  

        return padded_batch

    padded_center = pad_batch("center", ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask'])
    padded_center['labels'] = torch.stack([torch.from_numpy(el["center"]["labels"]) for el in batch])

    padded_left = pad_batch("left", ['input_ids', 'token_type_ids', 'attention_mask'])
    padded_right = pad_batch("right", ['input_ids', 'token_type_ids', 'attention_mask'])

    return {
        "left": padded_left,
        "center": padded_center,
        "right": padded_right
    }


data_filter = DataFilter(310)

mem_acc = Accumulator(rmt, decay=0.9)
pred_acc = Accumulator(predictor, decay=0.9)

memup_iter_acc = MemoryRollout[Dict[str, torch.Tensor]](
    steps=1000,
    memory=MemUpMemoryRMT(mem_acc.get_module()),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)
mem_acc.get_module().eval()
pred_acc.get_module().eval()

simple_corr = []

data_path = "/home/jovyan/human_test.h5"
train_dataset = EnformerDataset(tokenizer, data_path)

print(f'len(train_dataset): {len(train_dataset)}')

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=40, num_workers=5, collate_fn=collate_fn)


for it, batch in enumerate(train_dataloader):
    
    info = {}
    done = False
    print()
    state = rmt.init_state(batch["center"]["labels"].shape[0])

    train_set = []
    train_state = None

    with torch.no_grad():
        context_collector, last_state, _, _ = memup_iter_acc.forward(batch, state, {}, ContextCollector())
        train_state = last_state
        context = context_collector.result()
        print("context", context.shape)
        if context.shape[1] != 896:
            continue

        predictions = []

        for j in range(0, 896, 28):
            mask = torch.ones(context.shape[0], 28, dtype=torch.bool).cuda()
            pred_j = pred_acc(context[:, j:j+28].cuda(), last_state.cuda(), mask).cpu()
            # tg_j = batch["center"]["labels"][:, j:j+28]
            predictions.append(pred_j)

        predictions = torch.cat(predictions, 1)
        print(predictions.shape)

        pearson_corr_coef.update(predictions, batch["center"]["labels"])
        p_corr = pearson_corr_coef.compute().mean().item()
        print("pearson_corr_coef", p_corr)

        simple_corr.append(PearsonCorrLoss()(predictions.reshape(-1, 5313), batch["center"]["labels"].reshape(-1, 5313)).item())
        print("simple_corr_coef", sum(simple_corr) / len(simple_corr))

    

