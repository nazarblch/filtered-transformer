import json
import logging
import random
import sys
import os
import time
from typing import Dict

from regex import P

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
from examples.enformer_3.data import EnformerDataset
from examples.enformer_3.modules import BertForEnformer, ContextCollector, DataCollectorTrain, DataFilter, MemUpMemoryImpl
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/')

def collate_fn(batch):
    pad_token_ids = {'input_ids': tokenizer.pad_token_id, 'token_type_ids': 0, 'attention_mask': 0, 'bins_mask': 0, 'labels': -1}

    def pad_batch(feature_keys):

        padded_batch = {}
        
        for k in feature_keys:
            padded_batch[k] = pad_sequence(
                [torch.from_numpy(el[k]) for el in batch], 
                batch_first=True, 
                padding_value=pad_token_ids[k]
            )

        return padded_batch

    padded = pad_batch(['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask', 'labels'])

    padded["labels_mask"] = pad_sequence(
        [torch.ones(el["labels"].shape[0]) for el in batch],
        batch_first=True,
        padding_value=0
    ).type(torch.bool)

    for i in range(padded['input_ids'].shape[0]):
        assert padded['input_ids'][i][padded["bins_mask"][i]].shape[0] == padded['labels'][i][padded["labels_mask"][i]].shape[0]
    
    return padded


data_path = "/home/jovyan/human_train.h5"
train_dataset = EnformerDataset(tokenizer, data_path)

print(f'len(train_dataset): {len(train_dataset)}')

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, pin_memory=True, num_workers=8, collate_fn=collate_fn)


model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json')
model_cfg.num_labels = EnformerDataset.TG_COUNT
model = BertForEnformer(config=model_cfg, tokenizer=tokenizer)
predictor = Predictor(model_cfg, 1)

# bert = BertForTokenClassification(config=model_cfg)
# ckpt_path = '/home/jovyan/splice/model_1900000.pth'
# checkpoint = torch.load(ckpt_path, map_location='cpu')
# bert.load_state_dict(checkpoint["model_state_dict"], strict=False)
# weights = bert.bert.state_dict()
# model.bert.load_state_dict(weights)

weights = torch.load("/home/jovyan/enformer_10.0.pt", map_location="cpu")
model.load_state_dict(weights["mem"])
predictor.load_state_dict(weights["pred"])

model = model.cuda()
predictor = predictor.cuda()
model.train()
predictor.train()


data_filter = DataFilter(511)

memup_iter = MemoryRollout[Dict[str, torch.Tensor]](
    steps=2,
    memory=MemUpMemoryImpl(model),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)


optimizer = AdamW([
    {"params": model.bert.parameters(), "lr": 5e-6},
    {"params": model.encoder.parameters(), "lr": 2e-5},
    {"params": predictor.parameters(), "lr": 2e-5},
] , weight_decay=1e-5)


loss_modules = [
        LossModule(nn.PoissonNLLLoss(log_input=False), "poisson", 1.0),
        LossModule(PearsonCorrLoss(), "pearson corr", 0.0),
]

def pred_loss(state, out, target, mask):
        target = target.cuda()
        pred = predictor(out, state, mask)

        losses = {}
        sum_loss = 0

        for m in loss_modules:
            loss_item = m.module(pred[mask], target[mask])
            sum_loss = sum_loss + loss_item * m.coefficient
            losses[m.name] = loss_item.item()

        return sum_loss, losses


writer = SummaryWriter("/home/jovyan/pomoika/enformer10.1")
global_step = 0


for _ in range(10):

    for it, batch in enumerate(train_dataloader):

        if it % 50 == 0:
            pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)

        model.train()
        predictor.train()
        
        info = {}
        done = False
        print()
        state = torch.zeros(batch["labels"].shape[0], 100, model_cfg.hidden_size, device=torch.device("cuda:0"))
        labels_mask = batch["labels_mask"].cuda()

        with torch.no_grad():
            context_collector, last_state, _, _ = memup_iter.forward(batch, state, {}, ContextCollector(), steps=1000)
            context = context_collector.result()
            print("context", context.shape)
            last_state = last_state.cuda()
            context = context.cuda()
            
            pearson_corr_coef.reduce_dims=(0,)
            prediction = predictor(context, last_state, labels_mask).cpu()
            pearson_corr_coef.update(prediction[batch["labels_mask"]], batch["labels"][batch["labels_mask"]])
                
            p_corr = pearson_corr_coef.compute().mean().item()
            print("pearson_corr_coef", p_corr)
            writer.add_scalar(f"train/pearson_corr_coef", p_corr, global_step)

            labels = batch["labels"].cuda()


        while not done:
            global_step += 1

            optimizer.zero_grad()

            data_collector, state, info, done = memup_iter.forward(batch, state, info, DataCollectorTrain())
            out_seq, state_seq, mask_seq = data_collector.result()
            
            s0 = random.choice(list(state_seq) + [last_state])
            
            for m, o in zip(mask_seq, out_seq):
                context[m] = o
            
            loss, losses = pred_loss(s0, context, labels, labels_mask)
            context = context.detach()
            print(it, loss.item())
            print(losses)
            loss.backward()
            optimizer.step()

            for name, val in losses.items():
                writer.add_scalar(f"train/{name}", val, global_step)

            if global_step % 1000 == 0:
                    torch.save({
                        "mem": model.state_dict(),
                        "pred": predictor.state_dict(),
                        # "mem_acc": mem_acc.get_module().state_dict(),
                        # "pred_acc": pred_acc.get_module().state_dict()
                    }, "/home/jovyan/enformer_10.0.pt")


            

           


             

