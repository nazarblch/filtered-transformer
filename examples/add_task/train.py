import torch
from torch.utils.tensorboard import SummaryWriter
import time
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data_filters.sliding_window import SlidingWindowFilterTuple
from metrics.mse import MSEMetric
from modules import Predictor, DataType, MemUpMemoryImpl, SeqDataFilterImpl, DataCollectorEvalWithState, DataCollectorTrain
from data import AddTask
from memup.accumulator import Accumulator
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, State, Info, Done, InfoUpdate, \
    MemoryRolloutWithLoss, MemoryRollout, DataCollectorEmpty
from memup.loss import PredictorLossWithContext, LossModule, EvalLoss, TOS, PT, EvalLossWithMask
from memup.preproc import ContextPreprocessor, NStepUpdate, IncrementStep, ErrorPreprocessor, TargetsSampler, \
    TailTargets, select_by_index
from common_modules.pos_encoding import EmbedWithPos, LinearEmbedWithPos
from common_modules.transformers import BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer, TorchRecurrentTransformer, TorchRecurrentNN


mem_transformer = TorchRecurrentTransformer(128, 4, 3, 512, dropout=0.1).cuda()
embed = LinearEmbedWithPos(2, 128, 5.0).cuda()
predictor = Predictor().cuda()

seq_length = 500
rollout = 50
state_length = 20
writer = SummaryWriter(f"/home/slavic/pomoika/add_{seq_length}_{time.time()}")

train_loader = DataLoader(AddTask(10000, seq_length), shuffle=True, batch_size=128)
test_loader = DataLoader(AddTask(1000, seq_length), shuffle=False, batch_size=250)

opt = torch.optim.Adam([
    {"params": mem_transformer.parameters(), "lr": 5e-5},
    {"params": embed.parameters(), "lr": 5e-5},
    {"params": predictor.parameters(), "lr": 5e-5}
])

mem_acc = Accumulator(mem_transformer, decay=0.9)
pred_acc = Accumulator(predictor, decay=0.9)
embed_acc = Accumulator(embed, decay=0.9)

data_filter = SlidingWindowFilterTuple[DataType](rollout, padding=0, skip_fields={"length"})

memup_iter = MemoryRolloutWithLoss[DataType, TOS](
    steps=2,
    memory=MemUpMemoryImpl(embed, mem_transformer),
    loss=PredictorLossWithContext(predictor, [LossModule(nn.MSELoss(), "MSE", 1.0)], cur_step_loss_coef=10),
    data_filter=data_filter,
    info_update=[
        IncrementStep(),
        NStepUpdate(ContextPreprocessor(MemUpMemoryImpl(embed_acc.get_module(), mem_acc.get_module()), data_filter), 200),
        NStepUpdate(ErrorPreprocessor(pred_acc.get_module(), nn.MSELoss(reduction="none"), lambda data: data.y), 200),
        NStepUpdate(TargetsSampler((1, 10), lambda data: data.y), 4, offset=0)
    ]
)

memup_iter_eval = MemoryRollout[DataType](
    steps=1000,
    memory=MemUpMemoryImpl(embed_acc.get_module(), mem_acc.get_module()),
    data_filter=data_filter,
    info_update=[
        IncrementStep()
    ]
)

eval_loss = EvalLossWithMask([MSEMetric()])


@torch.no_grad()
def eval(i):

    print("evaluate")
    mem_transformer.eval()
    predictor.eval()

    for x, y, m in test_loader:
        state2 = torch.zeros(x.shape[0], state_length, 128).cuda()

        _, last_state, _, _ = memup_iter_eval.forward(DataType(x, y, m, x.shape[1]), state2, {}, DataCollectorEmpty())
        collector, _, _, _ = memup_iter_eval.forward(DataType(x, y, m, x.shape[1]), state2, {}, DataCollectorEvalWithState(predictor, last_state))
        info2 = {"mask": m}
        eval_loss.forward(collector, info2)

        print(info2["metrics"])
        for name, val in info2["metrics"].items():
            writer.add_scalar(f"eval/{name} last state", val, i)

    mem_transformer.train()
    predictor.train()


for i in range(1000):
    print("epoch", i)
    eval(i)

    last_info = {}

    for x, y, m in train_loader:
        print()

        state = torch.zeros(x.shape[0], state_length, 128).cuda()
        T = x.shape[1]
        done = False
        info = {}

        while not done:

            opt.zero_grad()

            loss, state, info, done = memup_iter.forward(DataType(x, y, m, T), state, info, DataCollectorTrain())
            last_info = info['losses']

            loss.backward()
            opt.step()

        mem_acc.accumulate()
        pred_acc.accumulate()
        embed_acc.accumulate()

        print(last_info)
        for name, val in last_info.items():
            writer.add_scalar(f"train/{name}", val, i)

