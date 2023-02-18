from itertools import chain
import random
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from data import EnformerDataset
from data_filters.top_errors import InputTarget
from omegaconf import OmegaConf
from memup.accumulator import Accumulator 
from collections import namedtuple
from typing import Tuple, Optional
from torch import Tensor
from common_modules.transformers import BertRecurrentTransformerWithTokenizer
from memup.base import MemUpMemory, State, DataCollectorReplace, MemoryOut, DataCollectorAppend
from memup.loss import TS, TOS, PredictorLossWithContext
from torch import nn
from data_filters.sliding_window import SlidingWindowFilterTuple
from memup.loss import PredictorLoss, LossModule, PredictorLossStateOnly, EvalLossStateOnly
from memup.preproc import IncrementStep, select_by_index
from common_modules.transformers import BertRecurrentTransformerWithTokenizer, BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer
from transformers import AutoTokenizer
from gena_lm.modeling_bert import BertModel
from memup.base import MemoryRollout
import torch
from sklearn.decomposition import PCA
from modules import  DataCollectorTrain, DataType, MemUpMemoryImpl, PearsonCorrLoss, Predictor


conf = OmegaConf.load(os.path.dirname(__file__) + '/config.yaml')


train_data = EnformerDataset(os.path.join(conf.data.path, conf.data.train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=20)


rollout = conf.model.rec_block_size
state_length = conf.model.state_size
torch.cuda.set_device(conf.device)
data_filter = SlidingWindowFilterTuple[DataType](rollout, pad_fields={"text"}, padding=conf.model.rec_block_padding, skip_fields={"target", "length", "coord"})


tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert: BertModel = BertModel.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert.train()
mem_transformer = BertRecurrentTransformerWithTokenizer(bert, tokenizer, conf.model.max_token_length, 6, 4, bert.config.hidden_size * 2).cuda()


predictor = Predictor(bert).cuda()


opt = torch.optim.Adam([
    {"params": mem_transformer.bert.parameters(), "lr": 8e-6},
    {"params": predictor.parameters(), "lr": 2e-5},
    {"params": mem_transformer.encoder.parameters(), "lr": 2e-5},
])

memup_iter = MemoryRollout[DataType](
    steps=3,
    memory=MemUpMemoryImpl(mem_transformer),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)


predictor_loss = PredictorLoss(predictor, [
    LossModule(nn.PoissonNLLLoss(log_input=False), "poisson", 1.0),
    LossModule(PearsonCorrLoss(), "pearson corr", 2.0),
])


writer = SummaryWriter(conf.data.logsdir)


def train_one_epoch(memup_iter, train_loader, global_step):


    for text, target, coords in train_loader:
        print("step", global_step)

        state = torch.zeros(target.shape[0], state_length, bert.config.hidden_size, device=torch.device("cuda"))
        T = len(text[0]) - EnformerDataset.PAD
        done = False
        info = {}
        T1 = 10
        target = target[:, 0:T1]
        coords = coords[:, 0:T1]

            
        mem_transformer.train()
        predictor.train()

        while not done:
            global_step += 1

            data_collector, state, info, done = memup_iter.forward(DataType(text, target, coords, T), state, info, DataCollectorTrain())

            opt.zero_grad()
            _, state_seq = data_collector.result()
            context = state_seq[-1][:, : T1 * 4]
            loss, losses = predictor_loss.loss(torch.cat(state_seq), context, target)
            print(global_step, losses)
            
            for name, val in losses.items():
                writer.add_scalar(f"train/{name}", val, global_step)

            loss.backward()
            opt.step()

        # mem_acc.accumulate()

    return global_step


global_step = 0

for _ in range(100):
    global_step = train_one_epoch(memup_iter, train_loader, global_step)

