import json
import logging
import random
import sys
import os
import time
from typing import Dict

from examples.enformer_2.modules import Predictor
from examples.spliceai.modules import BertForTokenClassification
sys.path.append(os.getcwd())

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
from examples.enformer_3.data import EnformerDataset, TestEnformerDataset
from examples.enformer_3.modules import BertForEnformer, ContextCollector, DataCollectorTrain, DataFilter, MemUpMemoryImpl
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/')

def collate_fn(batch_with_chunks):
    pad_token_ids = {'input_ids': tokenizer.pad_token_id, 'token_type_ids': 0, 'attention_mask': 0, 'bins_mask': 0, 'labels': -1}

    def pad_batch(batch, feature_keys):
        padded_batch = {}
        for k in feature_keys:
            padded_batch[k] = pad_sequence(
                [torch.from_numpy(el[k]) for el in batch], 
                batch_first=True, 
                padding_value=pad_token_ids[k]
            )
        return padded_batch

    chunks = []
    emply = {k: v[:0] for k, v in batch_with_chunks[0]["chunks"][0].items()}

    for pos in range(max([len(el["chunks"]) for el in batch_with_chunks])):
        batch = []
        for el in batch_with_chunks:
            if len(el["chunks"]) > pos:
                batch.append(el["chunks"][pos])
            else:
                batch.append(emply)
         
        padded = pad_batch(batch, ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask'])

        padded["labels_mask"] = pad_sequence(
            [torch.ones(el["bins_mask"].astype(np.int32).sum()) for el in batch],
            batch_first=True,
            padding_value=0
        ).type(torch.bool)

        chunks.append(padded)

    labels = torch.stack([torch.from_numpy(el["labels"]) for el in batch_with_chunks])

    return chunks, labels


data_path = "/home/jovyan/human_test.h5"
train_dataset = TestEnformerDataset(tokenizer, data_path)

print(f'len(train_dataset): {len(train_dataset)}')

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=32, pin_memory=True, num_workers=4, collate_fn=collate_fn)

model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json')
model_cfg.num_labels = EnformerDataset.TG_COUNT
model = BertForEnformer(config=model_cfg, tokenizer=tokenizer)
predictor = Predictor(model_cfg, 1)

weights = torch.load("/home/jovyan/enformer_10.0.pt", map_location="cpu")
model.load_state_dict(weights["mem"])
predictor.load_state_dict(weights["pred"])

model = model.cuda()
predictor = predictor.cuda()
model.eval()
predictor.eval()


data_filter = DataFilter(511)

memup_iter = MemoryRollout[Dict[str, torch.Tensor]](
    steps=2,
    memory=MemUpMemoryImpl(model),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)


with torch.no_grad():

    for it, (chunks, labels) in enumerate(train_dataloader):

        preds = []
        masks = []
        print("chunks count", len(chunks))
        print(chunks[0]['input_ids'].shape)

        state = torch.zeros(labels.shape[0], 100, model_cfg.hidden_size, device=torch.device("cuda:0"))

        for batch in chunks:

            context_collector, last_state, _, _ = memup_iter.forward(batch, state, {}, ContextCollector(), steps=100)
            context = context_collector.result()
            last_state = last_state.cuda()
            context = context.cuda()
            prediction = predictor(context, last_state, batch["labels_mask"].cuda()).cpu()

            preds.append(prediction)
            masks.append(batch["labels_mask"])
            # print(batch["labels_mask"][0])

        preds = torch.cat(preds, 1)
        masks = torch.cat(masks, 1)

        preds = preds[masks].reshape(labels.shape[0], 896, -1)
        print(preds.shape)

        pearson_corr_coef.update(preds, labels)
        p_corr = pearson_corr_coef.compute().mean().item()
        print("pearson_corr_coef", p_corr)

            
        


    