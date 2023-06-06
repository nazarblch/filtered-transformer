import json
import logging
import random
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
from examples.enformer_2.modules import BertForEnformer, ContextCollector, DataCollectorTrain, DataFilter, LinearPredictor, MemUpMemoryImpl, MemUpMemoryRMT, Predictor
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel, BertForMaskedLM
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from common_modules.rmt import RecurrentTransformerWithStateEmbedding


torch.cuda.set_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained('/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/')
model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln-lastln.json')
model_cfg.num_labels = EnformerDataset.TG_COUNT


rmt = RecurrentTransformerWithStateEmbedding(BertForMaskedLM(model_cfg), 10, tokenizer)

# weights = torch.load("/home/jovyan/rmt/model_best.pth", map_location="cpu")
# missing_keys, unexpected_keys = rmt.load_state_dict(weights["model_state_dict"], strict=False)
# print("=====================")
# print(missing_keys)
# print("=====================")
# print(unexpected_keys)
# print("=====================")

rmt = rmt.cuda()
rmt.train()

predictor = LinearPredictor(model_cfg).cuda()
predictor.train()

weights = torch.load("/home/jovyan/enformer_5.3.pt", map_location="cpu")
rmt.load_state_dict(weights["mem_acc"])
# predictor.load_state_dict(weights["pred_acc"])

optimizer = AdamW([
    {"params": rmt.parameters(), "lr": 1e-5},
    {"params": predictor.parameters(), "lr": 1e-5},
] , weight_decay=1e-5)


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


data_path = "/home/jovyan/human_train.h5"
train_dataset = EnformerDataset(tokenizer, data_path)

print(f'len(train_dataset): {len(train_dataset)}')

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10, pin_memory=True, num_workers=4, collate_fn=collate_fn)

data_filter = DataFilter(500)

memup_iter = MemoryRollout[Dict[str, torch.Tensor]](
    steps=6,
    memory=MemUpMemoryRMT(rmt),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)


loss_modules = [
        LossModule(nn.PoissonNLLLoss(log_input=False), "poisson", 1.0),
        LossModule(PearsonCorrLoss(), "pearson corr", 0.0),
]

# def pred_loss( state, out, target, mask):
#         target = target.cuda()
#         # N = state.shape[0] // target.shape[0]
#         # out = torch.cat([out] * N, 0)
#         # target = torch.cat([target] * N, 0)
#         pred = predictor(out, state)

#         losses = {}
#         sum_loss = 0

#         for m in loss_modules:
#             loss_item = m.module(pred.reshape(-1, 5313), target.reshape(-1, 5313))
#             if torch.any(mask):
#                 loss_item += m.module(pred[mask], target[mask]) * 0.5
#             sum_loss = sum_loss + loss_item * m.coefficient
#             losses[m.name] = loss_item.item()

#         return sum_loss, losses


# mem_acc = Accumulator(rmt, decay=0.9)
# pred_acc = Accumulator(predictor, decay=0.9)

# memup_iter_acc = MemoryRollout[Dict[str, torch.Tensor]](
#     steps=1000,
#     memory=MemUpMemoryRMT(mem_acc.get_module()),
#     data_filter=data_filter,
#     info_update=[IncrementStep()]
# )

writer = SummaryWriter("/home/jovyan/pomoika/enformer5.4")
global_step = 0


for _ in range(10):

    for it, batch in enumerate(train_dataloader):

        rmt.train()
        predictor.train()
        
        info = {}
        done = False
        print()
        B = batch["center"]["input_ids"].shape[0]
        state = rmt.init_state(B).cuda()

        with torch.no_grad():
            context_collector, _, _, _ = memup_iter.forward(batch, state, {}, ContextCollector(), steps=1000)
            context = torch.cat([c for c, _ in context_collector.collection], 1)
            c_mask = torch.cat([m for _, m in context_collector.collection], 1)
            B, _, D = context.shape
            context = context[c_mask].reshape(B, 896, D).cuda()
            print("context", context.shape)
            # last_state = last_state.cuda()

            pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)
            prediction = predictor(context).cpu()
            tg = batch["center"]["labels"]
            pearson_corr_coef.update(prediction, tg)
                
            p_corr = pearson_corr_coef.compute().mean().item()
            print("pearson_corr_coef", p_corr)
            writer.add_scalar(f"train/pearson_corr_coef_all", p_corr, global_step)

        labels = batch["center"]["labels"]

        while not done:
            global_step += 1

            optimizer.zero_grad()

            data_collector, state, info, done = memup_iter.forward(batch, state, info, DataCollectorTrain())
            out_seq, state_seq, mask_seq, global_mask_seq = data_collector.result()
            global_mask = torch.zeros(B, 896 + 2 * 80, device=state.device).type(torch.bool)
            for m in global_mask_seq:
                global_mask += m

            # s0 = random.choice(list(state_seq) + [last_state])

            out = torch.cat(list(filter(lambda o: o is not None and o.shape[1] > 0, out_seq)), 1) 
            mask = torch.cat(list(filter(lambda o: o is not None and o.shape[1] > 0, mask_seq)), 1)
            assert out.shape[1] == mask.shape[1]

            # context[global_mask] = out[mask]
            
            # loss, losses = pred_loss(s0, context, labels, global_mask[:, 80:-80])
            # context = context.detach()

            losses = {}
            sum_loss = 0

            if torch.any(mask):
                for m in loss_modules:
                    loss_item = m.module(predictor(out[mask]), labels[global_mask[:, 80:-80].cpu()].cuda())
                    sum_loss = sum_loss + loss_item * m.coefficient
                    losses[m.name] = loss_item.item()

                print(it, sum_loss.item())
                sum_loss.backward()
                optimizer.step()

            print(losses)

            for name, val in losses.items():
                writer.add_scalar(f"train/{name}", val, global_step)

            if global_step % 1000 == 0:
                    torch.save({
                        "mem": rmt.state_dict(),
                        "pred": predictor.state_dict(),
                        # "mem_acc": mem_acc.get_module().state_dict(),
                        # "pred_acc": pred_acc.get_module().state_dict()
                    }, "/home/jovyan/enformer_5.4.pt")

        # pearson_corr_coef = MeanPearsonCorrCoefPerChannel(5313)
        # prediction = predictor(context, last_state).cpu()
        # pearson_corr_coef.update(prediction, tg)
        # p_corr = pearson_corr_coef.compute().mean().item()
        # print("pearson_corr_coef", p_corr)

        # mem_acc.accumulate()
        # pred_acc.accumulate()
        
        

             

