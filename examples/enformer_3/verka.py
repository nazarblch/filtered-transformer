import json
import logging
import sys
import os
import time
from typing import Dict
sys.path.append("/home/jovyan/filtered-transformer/")
from tqdm import tqdm
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
from examples.enformer_3.data import EnformerDataset
from examples.enformer_3.modules import BertForEnformer
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn.utils.rnn import pad_sequence

torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/')


pad_token_ids = {'input_ids': tokenizer.pad_token_id, 'token_type_ids': 0, 'attention_mask': 0,
                    'bins_mask': 0, 'labels': 0}
pad_to_divisible_by = 2

def collate_fn(batch):
    feature_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask']
    padded_batch = {k: [] for k in feature_keys}
    max_seq_len = max([len(el['input_ids']) for el in batch])
    max_seq_len += (
        (pad_to_divisible_by - max_seq_len % pad_to_divisible_by)
        if max_seq_len % pad_to_divisible_by != 0
        else 0
    )
    for k in feature_keys:
        for i, el in enumerate(batch):
            dtype = batch[i][k].dtype
            pad = np.array([pad_token_ids[k]] * max(0, max_seq_len - len(el[k])), dtype=dtype)
            padded_batch[k] += [np.concatenate([batch[i][k], pad])]

    max_labels_len = max([len(el['labels']) for el in batch])
    padded_batch['labels'] = []
    padded_batch['labels_mask'] = torch.ones((len(batch), max_labels_len), dtype=torch.bool)
    for i, el in enumerate(batch):
        el = el['labels']
        pad = np.ones((max(0, max_labels_len - len(el)), el.shape[-1])) * pad_token_ids['labels']
        padded_batch['labels'] += [np.concatenate([batch[i]['labels'], pad])]
        padded_batch['labels_mask'][i, len(el):] = 0

    for k in padded_batch:
        padded_batch[k] = torch.from_numpy(np.stack(padded_batch[k]))

    return padded_batch


# get train dataset
train_dataloader = None
valid_dataloader = None
test_dataloader = None

test_data_path = "/home/jovyan/human_test.h5"
test_dataset = EnformerDataset(tokenizer, test_data_path, max_seq_len=512, bins_per_sample=24)
test_dataloader = DataLoader(test_dataset, batch_size=40, num_workers=5, collate_fn=collate_fn, shuffle=False)

# define model
model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json')
model_cfg.num_labels = EnformerDataset.TG_COUNT
model = BertForEnformer(config=model_cfg)

checkpoint = torch.load("/home/jovyan/model_best.pth", map_location='cpu')
missing_k, unexpected_k = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
if len(missing_k) != 0:
    print(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
if len(unexpected_k) != 0:
    print(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')


def batch_transform_fn(batch):
    return {
        'input_ids': batch['input_ids'],
        'token_type_ids': batch['token_type_ids'],
        'attention_mask': batch['attention_mask'],
        'labels': batch['labels'],
        'labels_mask': batch['labels_mask'],
        'bins_mask': batch['bins_mask'],
    }

model = model.cuda()
model = model.eval()

datasets = {'test': test_dataloader}
for sn in datasets:
    dl = datasets[sn]
    if dl is not None:
        corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=EnformerDataset.TG_COUNT)
        with torch.no_grad():
            for batch in tqdm(dl, desc=sn):
                batch = batch_transform_fn(batch)
                batch = {k: batch[k].cuda() for k in batch}
                output = model(**batch)
                pred = torch.nn.functional.softplus(output['logits'].detach())
                labels = batch['labels'][batch['labels_mask']]
                corr_coef.update(preds=pred.cpu(), target=labels.cpu())
                print(f'{sn} corr_coef: {corr_coef.compute().mean()}')


