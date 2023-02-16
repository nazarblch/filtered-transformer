from itertools import chain
import random
import time
from data import EnformerDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
sys.path.append("/home/buzun/filtered-transformer")
from data_filters.top_errors import InputTarget
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
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification
from memup.base import MemoryRollout
import torch
from sklearn.decomposition import PCA
from modules import  DataCollectorTrain, DataType, MemUpMemoryImpl, PearsonCorrLoss, Predictor, SeqDataFilterImpl


train_data = EnformerDataset("/mnt/nfs_dna/DNALM/downstream_tasks/enformer/human/h5/human_train.h5")
train_loader = DataLoader(train_data, shuffle=True, batch_size=20)


rollout = 800
state_length = 100
torch.cuda.set_device("cuda:1")
data_filter = SeqDataFilterImpl(rollout, padding=200)


tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert: BertModel = BertModel.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert.train()
mem_transformer = BertRecurrentTransformerWithTokenizer(bert, tokenizer, 270, 4, 4, bert.config.hidden_size * 2).cuda()


predictor = Predictor(bert).cuda()


opt = torch.optim.Adam([
    {"params": mem_transformer.bert.parameters(), "lr": 8e-6},
    {"params": predictor.parameters(), "lr": 2e-5},
    {"params": mem_transformer.encoder.parameters(), "lr": 2e-5},
])

memup_iter = MemoryRollout[DataType](
    steps=2,
    memory=MemUpMemoryImpl(mem_transformer),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

mem_acc = Accumulator(mem_transformer, decay=0.9)

memup_iter_acc = MemoryRollout[DataType](
    steps=1000,
    memory=MemUpMemoryImpl(mem_acc.get_module()),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)


predictor_loss = PredictorLossWithContext(predictor, [
    LossModule(nn.PoissonNLLLoss(log_input=False), "poisson", 1.0),
    LossModule(PearsonCorrLoss(), "pearson corr", 10.0),
], cur_step_loss_coef=1)


class ContextCollector(DataCollectorAppend[DataType, Tuple[DataType, Tensor]]):

    def append(self, data: DataType, out: MemoryOut, state: State) -> None:
        if out is not None:
            # print("append context")
            self.collection.append((data, state.detach().cpu()))

    def apply(self, data: DataType, out: MemoryOut, state: State) -> None:
        pass


writer = SummaryWriter(f"/home/buzun/pomoika/enformer_{time.time()}")


def train_one_epoch(memup_iter, train_loader, global_step):


    for text, target, coords in train_loader:
        print("step", global_step)

        state = torch.zeros(target.shape[0], state_length, bert.config.hidden_size, device=torch.device("cuda"))
        T = len(text[0])
        done = False
        info = {}

        with torch.no_grad():
            mem_acc.get_module().eval()
            context_collector, _, _, _ = memup_iter_acc.forward(DataType(text, target, coords, T, []), state, {}, ContextCollector())
            # print("context", len(context_collector.collection))
            
        mem_transformer.train()
        predictor.train()

        while not done:
            global_step += 1

            data_collector, state, info, done = memup_iter.forward(DataType(text, target, coords, T, []), state, info, DataCollectorTrain())
            
            index = random.randint(1, len(context_collector.collection)-1)
            selected_data = context_collector.collection[index][0]
            selected_state = context_collector.collection[index - 1][1]

            opt.zero_grad()
            context, _ = memup_iter.memory.forward(selected_data, selected_state.cuda())
            loss = predictor_loss.forward(data_collector, info, InputTarget(context, selected_data.target, context.shape[1]))
            print(info["losses"])
            
            for name, val in info["losses"].items():
                writer.add_scalar(f"train/{name}", val, global_step)

            loss.backward()
            opt.step()

        mem_acc.accumulate()

    return global_step


global_step = 0

for _ in range(100):
    global_step = train_one_epoch(memup_iter, train_loader, global_step)

