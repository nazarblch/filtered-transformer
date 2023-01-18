from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets.enformer_h5 import EnformerDataset
from memup.accumulator import Accumulator
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, MemUpLossIterator, State, Info, Done, InfoUpdate
from memup.data_filters import SlidingWindowFilter
from memup.loss import PredictorLossWithContext, LossModule, EvalLoss
from memup.preproc import ContextPreprocessor, NStepUpdate, IncrementStep, ErrorPreprocessor, TargetsSampler
from metrics.pearson import PearsonCorrLoss, PearsonCorrMetric
from models.transformers import BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
bert: BertModel = BertModel.from_pretrained('AIRI-Institute/gena-lm-bert-base')
mem_transformer = BertRecurrentTransformerWithTokenizer(bert, tokenizer, 320, 8, 4, bert.config.hidden_size * 2).cuda()


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = RecurrentTransformerFromBert(bert, 8, 4, bert.config.hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(bert.config.hidden_size, bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(bert.config.hidden_size, 5313)
        )

    def forward(self, x, state):
        out = self.encoder.forward(x, state).out
        return self.head(out).abs()


predictor = Predictor().cuda()

opt = torch.optim.Adam([
    {"params": mem_transformer.bert.parameters(), "lr": 5e-6},
    {"params": mem_transformer.encoder.parameters(), "lr": 3e-5},
    {"params": predictor.parameters(), "lr": 3e-5}
])


writer = SummaryWriter(f"/home/jovyan/pomoika/enformer_{time.time()}")


device = torch.device("cuda")
BS = 1000
TOPK = 100


DataType = namedtuple("DataType", ["text", "target", "coords"])
DataTypeWithMemory = Tuple[DataType, Tensor, Tensor]


class SeqDataFilterImpl(SlidingWindowFilter[DataType]):

    def __init__(self):
        super().__init__(BS, padding=BS // 5)

    def filter_data(self, data: DataType, i1: int, i2: int, i1_pad: int, i2_pad: int) -> DataType:
        pad_text = [t[i1_pad:i2_pad] for t in data.text]
        filtered_target = data.target[(data.coords >= i1) * (i2 > data.coords)]\
            .view(data.target.shape[0], -1, data.target.shape[2])
        filtered_coords = data.coords[(data.coords >= i1) * (i2 > data.coords)]\
            .view(data.coords.shape[0], -1)

        return DataType(pad_text, filtered_target, filtered_coords)


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, mem_tr: BertRecurrentTransformerWithTokenizer):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: DataType, state: State) -> Tuple[Tensor, State]:
        os = self.mem_tr.forward(data.text, state)
        assert os.out.shape[1] >= data.target.shape[1]
        return os.out[:, os.out.shape[1] - data.target.shape[1]:], os.state


mem_acc = Accumulator(mem_transformer, decay=0.95)
pred_acc = Accumulator(predictor, decay=0.95)


memup_iter_eval = MemUpLossIterator[DataType](
    rollout=2000,
    memory=MemUpMemoryImpl(mem_transformer),
    loss=EvalLoss(predictor, [
        PearsonCorrMetric()
    ]),
    data_filter=SeqDataFilterImpl(),
    info_update=[
        IncrementStep()
    ]
)


memup_iter_with_extra_targets = MemUpLossIterator[DataType](
    rollout=2,
    memory=MemUpMemoryImpl(mem_transformer),
    loss=PredictorLossWithContext(predictor, [
        LossModule(nn.PoissonNLLLoss(log_input=False), "poisson", 1.0),
        LossModule(PearsonCorrLoss(), "pearson corr", 100.0),
    ], lambda data: data.target),
    data_filter=SeqDataFilterImpl(),
    info_update=[
        IncrementStep(),
        NStepUpdate(ContextPreprocessor(MemUpMemoryImpl(mem_acc.get_module()), SeqDataFilterImpl()), 200),
        NStepUpdate(ErrorPreprocessor(pred_acc.get_module(), nn.PoissonNLLLoss(log_input=False, reduction="none", full=True), lambda data: data.target), 200),
        NStepUpdate(TargetsSampler(TOPK, lambda data: data.target), 2, offset=1)
    ]
)


@torch.no_grad()
def evaluate(train_loader):
    text2, target2, coords2 = next(iter(train_loader))
    state2 = torch.zeros(target2.shape[0], 50, bert.config.hidden_size, device=device)
    info = {}
    _, _, info, _ = memup_iter_eval.forward(DataType(text2, target2, coords2), state2, info)
    return info


def train_one_epoch(memup_iter, train_loader, global_step):

    for text1, target1, coords1 in train_loader:
        print()

        eval_res = evaluate(train_loader)
        writer.add_scalar("eval/pearson corr coef 1", eval_res["pearson corr coef 1"], global_step)
        writer.add_scalar("eval/pearson corr coef 2", eval_res["pearson corr coef 2"], global_step)
        writer.add_scalar("eval/poisson errors", eval_res["poisson errors"], global_step)

        state = torch.zeros(target1.shape[0], 50, bert.config.hidden_size, device=device)
        done = False
        info = {}

        while not done:
            global_step += 1

            opt.zero_grad()
            loss, state, info, done = memup_iter.forward(DataType(text1, target1, coords1), state, info)
            assert loss is not None
            if loss is not None:
                loss.backward()
            opt.step()

            if global_step % 10 == 0:
                if "pearson_corr current" in info:
                    writer.add_scalar("pearson_corr/current", info["pearson_corr current"], global_step)
                    writer.add_scalar("poisson_nll/current", info["poisson_nll current"], global_step)
                writer.add_scalar("pearson_corr/selected", info["pearson_corr selected"], global_step)
                writer.add_scalar("poisson_nll/selected", info["poisson_nll selected"], global_step)
                writer.add_scalar("sum loss", info["sum loss"], global_step)

        mem_acc.accumulate()
        pred_acc.accumulate()

    return global_step


global_step = 0

for i in range(1000):
    print("epoch", i)
    if i % 133 == 108:
        continue
    train_data = EnformerDataset([f"/home/jovyan/enformer/h5/train_{i % 133}.h5"])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

    global_step = train_one_epoch(memup_iter_with_extra_targets, train_loader, global_step)

    if i % 133 == 0:
        torch.save({
            "mem": mem_transformer.state_dict(),
            "pred": predictor.state_dict()
        }, "/home/jovyan/enformer/model.pt")
