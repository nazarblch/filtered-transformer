import time
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from data_filters.sliding_window import SlidingWindowFilterTuple
from examples.promoters.data import Promoters
from torch.utils.data import Dataset, DataLoader
import torch

from memup.accumulator import Accumulator
from memup.loss import PredictorLoss, LossModule, PredictorLossStateOnly, EvalLossStateOnly
from memup.preproc import IncrementStep
from examples.promoters.modules import MemUpMemoryImpl, DataCollectorTrain, DataCollectorLastState
from common_modules.transformers import BertRecurrentTransformerWithTokenizer, BertClassifier
from transformers import AutoTokenizer
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification
from memup.base import MemoryRollout
from examples.promoters.modules import DataType
from metrics.accuracy import AccuracyMetric
from metrics.f1 import F1Metric

train_data = Promoters([
        "/home/slavic/PycharmProjects/promoters16/fold_1.csv",
        "/home/slavic/PycharmProjects/promoters16/fold_2.csv",
        "/home/slavic/PycharmProjects/promoters16/fold_5.csv"
])

test_data = Promoters([
        "/home/slavic/PycharmProjects/promoters16/fold_4.csv"
])

train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
test_loader = DataLoader(test_data, shuffle=False, batch_size=512)

rollout = 800
state_length = 50
data_filter = SlidingWindowFilterTuple[DataType](rollout, pad_fields={"text"}, padding=200, skip_fields={"target", "length"})

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert_model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert
mem_transformer = BertRecurrentTransformerWithTokenizer(bert_model, tokenizer, 270, 4, 3, bert_model.config.hidden_size * 2).cuda()
predictor = BertClassifier(2, bert_model.config, 4, 2, bert_model.config.hidden_size).cuda()

weights = torch.load("/home/slavic/PycharmProjects/promoter_e_1.pt")
mem_transformer.load_state_dict(weights["mem"])
predictor.load_state_dict(weights["pred"])

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

writer = SummaryWriter(f"/home/slavic/pomoika/promoters_{16000}_{time.time()}")


@torch.no_grad()
def eval(i):

    print("evaluate", i, len(test_data))
    mem_acc.get_module().eval()
    pred_acc.get_module().eval()

    all_pred = []
    all_labels = []

    n = 0
    for text, labels in test_loader:
        state2 = torch.zeros(labels.shape[0], state_length, bert_model.config.hidden_size, device=torch.device("cuda"))
        T = len(text[0])

        collector, _, _, _ = memup_iter_eval.forward(DataType(text, labels, T), state2, {}, DataCollectorLastState())
        info2 = {}
        eval_loss.forward(collector, info2)

        print(info2["metrics"])
        all_pred.append(info2["predictions"])
        all_labels.append(labels)

        # n += 1
        # if n > 30:
        #     break

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

        if global_step % 1000 == 0 and global_step > 0:
            eval(global_step)
            torch.save({
                "mem": mem_transformer.state_dict(),
                "pred": predictor.state_dict()
            }, "/home/slavic/PycharmProjects/promoter.pt")

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

