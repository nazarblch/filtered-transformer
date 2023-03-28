import json
import logging
import sys
import os
import time
from typing import Dict
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
from examples.enformer_2.data import EnformerDataset
from examples.enformer_2.modules import BertForEnformer, ContextCollector, DataCollectorTrain, DataFilter, MemUpMemoryImpl, Predictor
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter


torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/')
# bert: BertModel = BertModel.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json')
model_cfg.num_labels = EnformerDataset.TG_COUNT
model = BertForEnformer(config=model_cfg)
# weights = bert.state_dict()
# weights.pop("pooler.dense.weight")
# weights.pop("pooler.dense.bias")
# model.bert.load_state_dict(weights)

model = model.cuda()
model.train()

predictor = Predictor(model_cfg).cuda()
predictor.train()

weights = torch.load("/home/jovyan/enformer_3.pt", map_location="cpu")
model.load_state_dict(weights["mem"])
predictor.load_state_dict(weights["pred"])

optimizer = AdamW([
    {"params": model.bert.parameters(), "lr": 2e-5},
    {"params": model.encoder.parameters(), "lr": 5e-5},
    {"params": predictor.parameters(), "lr": 5e-5},
] , weight_decay=1e-5)

print("pad token id", tokenizer.pad_token_id)

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

    padded_left = pad_batch("left", ['input_ids', 'token_type_ids', 'attention_mask'])
    padded_right = pad_batch("right", ['input_ids', 'token_type_ids', 'attention_mask'])

    return {
        "left": padded_left,
        "center": padded_center,
        "right": padded_right
    }


data_path = "/home/jovyan/human_train.h5"
train_dataset = EnformerDataset(tokenizer, data_path)

print(f'len(train_dataset): {len(train_dataset)}')

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, pin_memory=True, num_workers=8, collate_fn=collate_fn)

data_filter = DataFilter(512)

memup_iter = MemoryRollout[Dict[str, torch.Tensor]](
    steps=2,
    memory=MemUpMemoryImpl(model),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)


predictor_loss = PredictorLossWithContext(predictor, [
        LossModule(nn.PoissonNLLLoss(log_input=False), "poisson", 1.0),
        LossModule(PearsonCorrLoss(), "pearson corr", 0.0),
], cur_step_loss_coef=1)


mem_acc = Accumulator(model, decay=0.9)
pred_acc = Accumulator(predictor, decay=0.9)

errors_filter = TopErrorsFilter(28, (25, 35), pred_acc, nn.PoissonNLLLoss(log_input=False, reduction="none"), is_random=True)

memup_iter_acc = MemoryRollout[Dict[str, torch.Tensor]](
    steps=1000,
    memory=MemUpMemoryImpl(mem_acc),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

writer = SummaryWriter("/home/jovyan/pomoika/enformer3.0")
global_step = 0


for _ in range(10):

    for it, batch in enumerate(train_dataloader):

        model.train()
        predictor.train()
        
        info = {}
        done = False
        print()
        state = torch.zeros(batch["center"]["labels"].shape[0], 300, model_cfg.hidden_size, device=torch.device("cuda:0"))

        with torch.no_grad():
            context_collector, last_state, _, _ = memup_iter_acc.forward(batch, state, {}, ContextCollector())
            context = torch.cat([c for c, _ in context_collector.collection], 1)
            c_mask = torch.cat([m for _, m in context_collector.collection], 1)
            B, _, D = context.shape
            context = context[c_mask].reshape(B, 896, D)
            print("context", context.shape)
            if context.shape[1] != 896:
                continue

            last_state = last_state.cuda()
            pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)
            for j in range(0, 896, 28):
                mask = torch.ones(B, 28, dtype=torch.bool).cuda()
                pred_j = pred_acc(context[:, j:j+28].cuda(), last_state, mask).cpu()
                tg_j = batch["center"]["labels"][:, j:j+28]
                pearson_corr_coef.update(pred_j, tg_j)
                
            p_corr = pearson_corr_coef.compute().mean().item()
            print("pearson_corr_coef", p_corr)
            writer.add_scalar(f"train/pearson_corr_coef_all", p_corr, global_step)

            selected_data, _ = errors_filter.forward(InputTarget(context, batch["center"]["labels"], 896), last_state, {})
        
        while not done:
            global_step += 1

            data_collector, state, info, done = memup_iter.forward(batch, state, info, DataCollectorTrain())
                
            optimizer.zero_grad()
            
            loss = predictor_loss.forward(data_collector, info, selected_data, last_state)
            print(it, loss.item())
            print(info["losses"])
            loss.backward()
            optimizer.step()

            for name, val in info["losses"].items():
                writer.add_scalar(f"train/{name}", val, global_step)

            if global_step % 1000 == 0:
                    torch.save({
                        "mem": model.state_dict(),
                        "pred": predictor.state_dict(),
                        "mem_acc": mem_acc.get_module().state_dict(),
                        "pred_acc": pred_acc.get_module().state_dict()
                    }, "/home/jovyan/enformer_3.pt")

        mem_acc.accumulate()
        pred_acc.accumulate()
        

             

