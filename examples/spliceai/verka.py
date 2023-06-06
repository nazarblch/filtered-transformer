import json
import logging
import os
import sys
from typing import Dict

from common_modules.pos_encoding import PositionalEncoding2
from data_filters.sliding_window import SlidingWindowFilterDict
from data_filters.top_errors import InputTargetMask, TopErrorsFilterWithMask
from memup.accumulator import Accumulator
from memup.base import MemoryRollout
from memup.loss import LossModule, PredictorLoss, PredictorLossWithContext
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
from modules import BertForSpliceAI, BertForTokenClassification, ContextCollector, Predictor, MemUpMemoryImpl, DataCollectorTrain, SpliceLoss, SpliceLossFlat
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/")

    data_path = "/home/jovyan/splice/dataset_test_0.csv.gz"
    # data_path = "/home/jovyan/splice/train.csv.gz"
    train_dataset = SpliceAIDataset(data_path, tokenizer, max_seq_len=510 * 6 + 1, targets_offset=5000, targets_len=5000)
    
    train_dataloader = DataLoader(train_dataset, batch_size=40, num_workers=4, shuffle=True)
    # define model
    model_cfg = AutoConfig.from_pretrained("/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json")
    model_cfg.num_labels = 3
    model_cfg.problem_type = 'multi_label_classification'
    model = BertForSpliceAI(config=model_cfg, tokenizer=tokenizer)
    print(model_cfg)
    
    model = model.cuda()
    predictor = Predictor(model_cfg).cuda()

    weights = torch.load("/home/jovyan/splice_2.pt", map_location="cpu")
    model.load_state_dict(weights["mem_acc"])
    predictor.load_state_dict(weights["pred_acc"])

    def batch_transform_fn(batch):
        return {
            'input_ids': batch['input_ids'].cuda()[:, 1:],
            'token_type_ids': batch['token_type_ids'].cuda()[:, 1:],
            'attention_mask': batch['attention_mask'].cuda()[:, 1:],
            'labels': batch['labels_ohe'].cuda()[:, 1:],
            'labels_mask': batch['labels_mask'].type(torch.bool).cuda()[:, 1:]
        }

    @torch.no_grad()
    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = data['labels'], torch.sigmoid(data['predictions'])
        y = y[data['labels_mask']]
        p = p[data['labels_mask']]
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


    data_filter = SlidingWindowFilterDict(510, pad_fields=set(), padding=0, skip_fields={"length"})
    pos_encoder = PositionalEncoding2(768, 0, 510 * 4)

    memup_iter_acc = MemoryRollout[Dict[str, torch.Tensor]](
        steps=1000,
        memory=MemUpMemoryImpl(model),
        data_filter=data_filter,
        info_update=[IncrementStep()]
    )

    model.eval()
    predictor.eval()

    mean_test = []

    for batch in train_dataloader:

        batch = batch_transform_fn(batch)
        B, T = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
        state = torch.zeros(B, 100, model_cfg.hidden_size, device=torch.device("cuda:0"))

        batch["length"] = T
        # batch["positions"] = pos_encoder.forward(torch.zeros(1, T, 768)).expand(B, T, 768)

        done = False
        info = {}

        with torch.no_grad():
            context_collector, last_state, _, _ = memup_iter_acc.forward(batch, state, {}, ContextCollector())
            context = context_collector.result()
            last_state = last_state.cuda()
          
            pred = []
            for dk in context:
                pred_t = predictor(dk.input.cuda(), last_state, dk.mask.cuda())
                pred.append(pred_t)

            pred = torch.cat(pred, 1).cpu()
            c_labels = torch.cat([dk.target for dk in context], 1)
            c_mask = torch.cat([dk.mask for dk in context], 1)

            metrics = metrics_fn({
                'labels': c_labels,
                'predictions': pred,
                'labels_mask': c_mask
            })
            print(metrics)

            if metrics['pr_auc_mean'] > 0.5:
                mean_test.append(metrics['pr_auc_mean'])

            print("mean test", sum(mean_test) / len(mean_test))
     
    
