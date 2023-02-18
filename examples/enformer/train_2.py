import time
from omegaconf import OmegaConf
from sklearn.compose import make_column_transformer
from torch import nn
import sys
import os
sys.path.append(os.getcwd())
from examples.enformer.data import EnformerDataset
from torch.utils.tensorboard import SummaryWriter
from data_filters.sliding_window import SlidingWindowFilterTuple
from torch.utils.data import Dataset, DataLoader
import torch
from metrics.f1 import F1Metric
from memup.accumulator import Accumulator
from memup.loss import PredictorLoss, LossModule, PredictorLossStateOnly, EvalLossStateOnly
from memup.preproc import IncrementStep
from common_modules.transformers import BertRecurrentTransformerWithTokenizer, BertClassifier
from transformers import AutoTokenizer
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification
from memup.base import MemoryRollout
from metrics.accuracy import AccuracyMetric
from examples.enformer.modules import DataCollectorTrainStateOnly, PearsonCorrLoss, PredictorStateOnly, MemUpMemoryImpl, DataType


conf = OmegaConf.load(os.path.dirname(__file__) + '/config.yaml')


train_data = EnformerDataset(os.path.join(conf.data.path, conf.data.train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=32)


rollout = conf.model.rec_block_size
state_length = conf.model.state_size
torch.cuda.set_device(conf.device)
data_filter = SlidingWindowFilterTuple[DataType](rollout, pad_fields={"text"}, padding=conf.model.rec_block_padding, skip_fields={"target", "length"})

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert_model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert
mem_transformer = BertRecurrentTransformerWithTokenizer(bert_model, tokenizer, conf.model.max_token_length, 4, 3, bert_model.config.hidden_size * 2).cuda()
predictor = PredictorStateOnly(bert_model, 10).cuda()

opt = torch.optim.Adam([
    {"params": mem_transformer.bert.parameters(), "lr": 4e-6},
    {"params": predictor.parameters(), "lr": 2e-5},
    {"params": mem_transformer.encoder.parameters(), "lr": 2e-5},
])

memup_iter = MemoryRollout[DataType](
    steps=2,
    memory=MemUpMemoryImpl(mem_transformer),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)


predictor_loss = PredictorLossStateOnly(predictor, [
    LossModule(nn.PoissonNLLLoss(log_input=False), "poisson", 1.0),
    LossModule(PearsonCorrLoss(), "pearson corr", 10.0),
])


writer = SummaryWriter(conf.data.logsdir)


def train_one_epoch(memup_iter, train_loader, global_step):

    # eval(global_step)

    last_info = {}

    for text, target, coords in train_loader:
        print()

        target = target[:, 10:20]

        if global_step % 1000 == 0 and global_step > 0:
            # eval(global_step)
            torch.save({
                "mem": mem_transformer.state_dict(),
                "pred": predictor.state_dict()
            }, conf.data.save_path)

        state = torch.zeros(target.shape[0], state_length, bert_model.config.hidden_size, device=torch.device(conf.device))
        T = len(text[0])
        done = False
        info = {}
   
        mem_transformer.train()
        predictor.train()

        while not done:
            global_step += 1
            data_collector, state, info, done = memup_iter.forward(DataType(text, target, coords, T, []), state, info, DataCollectorTrainStateOnly())
            opt.zero_grad()
            loss = predictor_loss.forward(data_collector, info)
            print(global_step, info["losses"])

            loss.backward()
            opt.step()

            if global_step % 5 == 0:
                for name, val in info["losses"].items():
                    writer.add_scalar(f"train/{name}", val, global_step)

    return global_step


global_step = 0

for it in range(1000):
    print("epoch", it)
    global_step = train_one_epoch(memup_iter, train_loader, global_step)

