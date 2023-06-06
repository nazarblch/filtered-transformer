import time
from pathlib import Path

from omegaconf import OmegaConf
from torch import nn
import sys
import os
sys.path.append(os.getcwd())
from torch.utils.tensorboard import SummaryWriter
from data_filters.sliding_window import SlidingWindowFilterTuple
from examples.promoters.data import Promoters
from torch.utils.data import Dataset, DataLoader
import torch
from metrics.f1 import F1Metric
from memup.accumulator import Accumulator
from memup.loss import PredictorLoss, LossModule, PredictorLossStateOnly, EvalLossStateOnly
from memup.preproc import IncrementStep
from examples.promoters.modules import MemUpMemoryImpl, DataCollectorTrain, DataCollectorLastState
from common_modules.transformers import BertRecurrentTransformerWithTokenizer, BertClassifier
from transformers import AutoTokenizer, AutoModel, AutoConfig
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification, BertForMaskedLM, BertConfig
from memup.base import MemoryRollout
from examples.promoters.modules import DataType
from metrics.accuracy import AccuracyMetric
from absl import flags

conf = OmegaConf.load(os.path.dirname(__file__) + '/config.yaml')


train_data = Promoters([os.path.join(conf.data.path, f) for f in conf.data.train])
test_data = Promoters([os.path.join(conf.data.path, conf.data.test)])

train_loader = DataLoader(train_data, shuffle=True, batch_size=conf.model.batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=conf.model.eval_batch_size)

rollout = conf.model.rec_block_size
state_length = conf.model.state_size
data_filter = SlidingWindowFilterTuple[DataType](rollout, pad_fields={"text"}, padding=conf.model.rec_block_padding, skip_fields={"target", "length"})

# tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
# bert_model: BertModel = BertForSequenceClassification.from_pretrained("AIRI-Institute/gena-lm-bert-base").bert
# checkpoint = torch.load("/home/slavic/PycharmProjects/bbert/pytorch_model.bin", map_location='cpu')
# bert_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
tokenizer = AutoTokenizer.from_pretrained("/tmp/pycharm_project_522/data/tokenizers/t2t_1000h_multi_32k")
config = BertConfig.from_pretrained('/tmp/pycharm_project_522/data/configs/L24-H1024-A16-V32k-preln-lastln.json')
bert_model = BertForSequenceClassification(config=config)
# ckpt_path = '/home/slavic/PycharmProjects/lbert/pytorch_model.bin'
# checkpoint = torch.load(ckpt_path, map_location='cpu')
# bert_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
bert_model = bert_model.bert

bert_model.train()
mem_transformer = BertRecurrentTransformerWithTokenizer(bert_model, tokenizer, conf.model.max_token_length, 4, 3, bert_model.config.hidden_size * 2).cuda()
predictor = BertClassifier(2, bert_model.config, 4, 2, bert_model.config.hidden_size).cuda()

weights = torch.load(conf.data.save_path)
mem_transformer.load_state_dict(weights["mem"])
# predictor.load_state_dict(weights["pred"])

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
        LossModule(nn.CrossEntropyLoss(), "CE", 1.0),
        LossModule(AccuracyMetric(), "Accuracy", 0.0)
])


mem_acc = Accumulator(mem_transformer, decay=0.9)
pred_acc = Accumulator(predictor, decay=0.9)

memup_iter_eval = MemoryRollout[DataType](
    steps=1000,
    memory=MemUpMemoryImpl(mem_acc.get_module()),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

eval_loss = EvalLossStateOnly(pred_acc.get_module(), [AccuracyMetric()])

writer = SummaryWriter(conf.data.logsdir)


@torch.no_grad()
def eval(i):

    print("evaluate", i, len(test_data))
    mem_acc.get_module().eval()
    pred_acc.get_module().eval()

    all_pred = []
    all_labels = []

    for text, labels in test_loader:
        state2 = torch.zeros(labels.shape[0], state_length, bert_model.config.hidden_size, device=torch.device("cuda"))
        T = len(text[0])

        collector, _, _, _ = memup_iter_eval.forward(DataType(text, labels, T), state2, {}, DataCollectorLastState())
        info2 = {}
        eval_loss.forward(collector, info2)

        print(info2["metrics"])
        all_pred.append(info2["predictions"])
        all_labels.append(labels)

    acc = AccuracyMetric()(torch.cat(all_pred, 0), torch.cat(all_labels, 0))
    f1 = F1Metric()(torch.cat(all_pred, 0), torch.cat(all_labels, 0))
    print("acc", acc, "f1", f1)
    writer.add_scalar("eval/Accuracy", acc, i)
    writer.add_scalar("eval/F1", f1, i)


def train_one_epoch(memup_iter, train_loader, global_step):

    # eval(global_step)

    last_info = {}

    for text, labels in train_loader:
        print()
        state = torch.zeros(labels.shape[0], state_length, bert_model.config.hidden_size, device=torch.device("cuda"))
        T = len(text[0])
        done = False
        info = {}

        mem_transformer.train()
        predictor.train()

        while not done:
            global_step += 1
            data_collector, state, info, done = memup_iter.forward(DataType(text, labels, T), state, info, DataCollectorTrain())
            opt.zero_grad()
            loss = predictor_loss.forward(data_collector, info)
            last_info = info["losses"]

            loss.backward()
            opt.step()

            if global_step % 1000 == 0:
                eval(global_step)
                torch.save({
                    "mem": mem_transformer.state_dict(),
                    "pred": predictor.state_dict()
                }, conf.data.save_path)

        print(global_step, last_info)

        for name, val in last_info.items():
            writer.add_scalar(f"train/{name}", val, global_step)

        mem_acc.accumulate()
        pred_acc.accumulate()

    return global_step


global_step = 0

for it in range(1000):
    print("epoch", it)
    global_step = train_one_epoch(memup_iter, train_loader, global_step)

