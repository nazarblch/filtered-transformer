import json
import logging
import os
import sys
from typing import Dict

from common_modules.pos_encoding import PositionalEncoding2
from data_filters.sliding_window import SlidingWindowFilterDict
from memup.base import MemoryRollout
from memup.preproc import IncrementStep

sys.path.append("/home/slavic/PycharmProjects/filtered-transformer")
from pathlib import Path
from examples.spliceai.data import SpliceAIDataset
import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from torch.nn import BCEWithLogitsLoss
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from sklearn.metrics import average_precision_score
import numpy as np
from torch.optim import AdamW
from modules import BertForSpliceAI, BertForTokenClassification, Predictor, MemUpMemoryImpl, DataCollectorTrain
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("/home/slavic/PycharmProjects/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/")

    data_path = "/home/slavic/Downloads/valid.csv.gz"
    train_dataset = SpliceAIDataset(data_path, tokenizer, max_seq_len=512,
                                    targets_offset=5000, targets_len=5000)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    # define model
    model_cfg = AutoConfig.from_pretrained("/home/slavic/PycharmProjects/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json")
    model_cfg.num_labels = 3
    model_cfg.problem_type = 'multi_label_classification'
    bert = BertForTokenClassification(config=model_cfg)
    model = BertForSpliceAI(config=model_cfg)

    ckpt_path = '/home/slavic/Downloads/model_1900000.pth'
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    bert.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.bert = bert.bert
    
    model = model.cuda()
    model.train()
    
    predictor = Predictor(model_cfg).cuda()
    predictor.train()

    # define optimizer
    optimizer = AdamW([
            {"params": model.bert.parameters(), "lr": 3e-5},
            {"params": model.encoder.parameters(), "lr": 5e-5},
            {"params": predictor.parameters(), "lr": 5e-5},
    ], weight_decay=1e-4)


    def loss_fn(logits, labels, labels_mask):
        bs, seq_len = logits.shape[:2]
        pos_weight = torch.tensor([1.0, 100.0, 100.0])
        pos_weight_seq = pos_weight.repeat(bs, seq_len, 1).cuda()
        loss_fct = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_seq)
        loss = loss_fct(logits, labels)
        loss = loss * labels_mask.unsqueeze(-1)
        loss = loss.sum() / labels_mask.sum() if labels_mask.sum() != 0.0 else torch.tensor(0.0, device=logits.device)
        return loss

    # label counts in test set: [8378616.,    9842.,   10258.])
    # upweight class 1 and 2

    def batch_transform_fn(batch):
        return {
            'input_ids': batch['input_ids'].cuda(),
            'token_type_ids': batch['token_type_ids'].cuda(),
            'attention_mask': batch['attention_mask'].cuda(),
            'labels': batch['labels_ohe'].cuda(),
            'labels_mask': batch['labels_mask'].cuda()
        }

    @torch.no_grad()
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        data['labels'] = batch['labels']
        data['predictions'] = output['logits'].detach()
        data['labels_mask'] = batch['labels_mask']
        return data

    @torch.no_grad()
    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = data['labels'], torch.sigmoid(data['predictions'])
        y = y[data['labels_mask'] == 1.0]
        p = p[data['labels_mask'] == 1.0]
        # compute pr-auc for each class independetly
        for label in [0, 1, 2]:
            y_label = y[:, label].cpu().numpy()
            p_label = p[:, label].cpu().numpy()
            if not np.isnan(p_label).any():
                pr_auc = average_precision_score(y_label, p_label, pos_label=1)
            else:
                pr_auc = np.nan
            # to be compatible with sklearn 1.1+
            metrics[f'pr_auc_{label}'] = pr_auc if not np.isnan(pr_auc) else 0.0
        metrics['pr_auc_mean'] = (metrics['pr_auc_1'] + metrics['pr_auc_2']) / 2
        return metrics


    data_filter = SlidingWindowFilterDict(256, pad_fields=set(), padding=0, skip_fields={"length"})
    pos_encoder = PositionalEncoding2(768, 0, 512)

    memup_iter = MemoryRollout[Dict[str, torch.Tensor]](
        steps=2,
        memory=MemUpMemoryImpl(model),
        data_filter=data_filter,
        info_update=[IncrementStep()]
    )
    
    writer = SummaryWriter("/home/slavic/pomoika/splice2.0")
    global_step = 0

    for batch in train_dataloader:
        B, T = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
        state = torch.zeros(B, 100, model_cfg.hidden_size, device=torch.device("cuda:0"))
        batch = batch_transform_fn(batch)

        batch["length"] = T
        batch["positions"] = pos_encoder.forward(torch.zeros(1, T, 768)).expand(B, T, 768)

        done = False
        info = {}
        pred = None

        while not done:
            global_step += 1

            optimizer.zero_grad()

            collector, state, info, done = memup_iter.forward(batch, state, info, DataCollectorTrain())
            target_seq, out_seq, state_seq, mask_seq = collector.result()
            pred_1 = predictor.forward(out_seq[0], state_seq[0], mask_seq[0])
            pred_2 = predictor.forward(out_seq[1], state_seq[1], mask_seq[1])
            loss_1 = loss_fn(pred_1, target_seq[0], mask_seq[0])
            loss_2 = loss_fn(pred_2, target_seq[1], mask_seq[1])

            loss = (loss_1 + loss_2) / 2
            pred = torch.cat([pred_1, pred_2], dim=1).detach()
            # pred = pred_1

            print("loss=", loss.item())
            writer.add_scalar("loss", loss.item(), global_step)

            loss.backward()
            optimizer.step()

        metrics = metrics_fn({
            'labels': batch['labels'],
            'predictions': pred,
            'labels_mask': batch['labels_mask']
        })
        print(metrics)
        writer.add_scalar('pr_auc_mean', metrics['pr_auc_mean'], global_step)
