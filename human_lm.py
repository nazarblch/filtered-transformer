import time
from copy import deepcopy
from itertools import chain
import pytorch_lightning as pl
import numpy as np
import torch
from gena_lm.modeling_bert import BertForSequenceClassification, BertModel, BertEncoder, BertForMaskedLM
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, BertConfig, DataCollatorForLanguageModeling, AdamW
import pandas as pd
from tqdm import tqdm
from datasets.gena import HumanDataset, HumanDataset2, MaskedLMDataset
from filter_model.base import FilterModel, FilteredRecurrentTransformer, NStepFilterObject
from filter_model.chunk_filter import BertChunkFilter
from filter_model.seq_filter import DictSeqFilterBidirectional, DictSeqFilter
from models.transformers import TransformerClassifier, BertClassifier, BertRecurrentTransformer, RecurrentOutputSeq, \
    RecurrentTransformerFromBert

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
train_dataset = MaskedLMDataset(
    "/home/nazar/PycharmProjects/GENA_LM/downstream_tasks/promoter_prediction/hg38_len_2000_promoters_dataset.csv",
    tokenizer
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

train_loader = DataLoader(
    train_dataset,
    batch_size=48,
    collate_fn=data_collator
)


bert_lm = BertForMaskedLM.from_pretrained('AIRI-Institute/gena-lm-bert-base')
mem_transformer = BertRecurrentTransformer(bert_lm.bert, 12, 4, bert_lm.config.hidden_size * 2)

transformer_rollout = FilteredRecurrentTransformer(
    mem_transformer,
    DictSeqFilter(50, "input_ids"),
    embedding=None,
    rollout=2
).cuda()

context_encoder = FilteredRecurrentTransformer(
    mem_transformer,
    DictSeqFilter(50, "input_ids"),
    embedding=None,
    rollout=100
).cuda()


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.cls = bert_lm.cls
        self.encoder = RecurrentTransformerFromBert(bert_lm.bert, 12, 2, bert_lm.config.hidden_size * 2)

    def forward(self, x, state):
        out = self.encoder.forward(x, state).out
        return self.cls(out)


predictor = Predictor().cuda()

opt = AdamW(chain(predictor.parameters(), transformer_rollout.parameters()), lr=1e-5)


def train_step(input_ids: Tensor, labels: Tensor):
    labels = labels[:, 1:].contiguous()
    last_label = torch.zeros(labels.shape[0], 1, dtype=labels.dtype, device=labels.device) - 100
    labels = torch.cat([labels, last_label], dim=1)

    s0 = torch.zeros(input_ids.shape[0], 50, bert_lm.config.hidden_size, device=input_ids.device)
    mlm_mask = torch.zeros_like(labels, dtype=torch.bool)
    mlm_mask[labels != 100] = True
    mask_lens = mlm_mask.type(torch.int64).sum(-1).detach().cpu().numpy().tolist()

    with torch.no_grad():
        context = next(context_encoder({"input_ids": input_ids}, s0)).get_cat_out()

    for os_seq in transformer_rollout({"input_ids": input_ids}, s0):

        opt.zero_grad()
        sequence_output = os_seq.get_cat_out()
        filter_mask = os_seq.get_sum_mask()

        context_with_out = torch.clone(context)
        context_with_out[filter_mask] = sequence_output.view(-1, sequence_output.shape[-1])

        context_with_out = pad_sequence(torch.split(context_with_out[mlm_mask], mask_lens, 0), batch_first=True)
        labels_select = pad_sequence(torch.split(labels[mlm_mask], mask_lens, 0), batch_first=True, padding_value=-100)

        prediction_scores = predictor(context_with_out, os_seq.states[-1])
        loss_fct = CrossEntropyLoss()
        lm_loss = loss_fct(prediction_scores.view(-1, bert_lm.config.vocab_size), labels_select.view(-1))
        lm_loss.backward()
        opt.step()

    return lm_loss.item()


writer = SummaryWriter(f"/home/nazar/pomoika/gena_2000_mlm_{time.time()}")
step = 0

for epoch in range(100):
    for data in tqdm(train_loader):
        input_ids, labels = data["input_ids"], data["labels"]
        loss = train_step(input_ids.cuda(), labels.cuda())
        writer.add_scalar("train loss", loss, step)
        step += 1

