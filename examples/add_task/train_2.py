import torch
from torch.utils.tensorboard import SummaryWriter
import time
from torch import Tensor, nn
from torch.utils.data import DataLoader
from metrics.mse import MSEMetric
from modules import Predictor, DataType, MemUpMemoryImpl, SeqDataFilterImpl2, DataCollectorEvalWithState, DataCollectorTrain, \
    PredictorFromState, DataCollectorTrainFromState, DataCollectorLastState
from data import AddTask
from memup.accumulator import Accumulator
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, State, Info, Done, InfoUpdate, \
    MemoryRolloutWithLoss, MemoryRollout, DataCollectorEmpty, DataCollectorReplace
from memup.loss import PredictorLossWithContext, LossModule, EvalLoss, TS, PT, EvalLossWithMask, PredictorLossStateOnly, EvalLossStateOnly
from memup.preproc import ContextPreprocessor, NStepUpdate, IncrementStep, ErrorPreprocessor, TargetsSampler, \
    TailTargets
from models.pos_encoding import EmbedWithPos, LinearEmbedWithPos
from models.transformers import BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer, TorchRecurrentTransformer, TorchRecurrentNN


mem_transformer = TorchRecurrentTransformer(128, 4, 3, 512, dropout=0.1).cuda()
embed = LinearEmbedWithPos(2, 128, 5.0).cuda()
predictor = PredictorFromState().cuda()

seq_length = 500
rollout = 50
state_length = 20
writer = SummaryWriter(f"/home/jovyan/pomoika/add_{seq_length}_{time.time()}")

train_loader = DataLoader(AddTask(10000, seq_length), shuffle=True, batch_size=128)
test_loader = DataLoader(AddTask(1000, seq_length), shuffle=False, batch_size=250)

opt = torch.optim.Adam([
    {"params": mem_transformer.parameters(), "lr": 5e-5},
    {"params": embed.parameters(), "lr": 5e-5},
    {"params": predictor.parameters(), "lr": 5e-5}
])


memup_iter = MemoryRolloutWithLoss[DataType, TS](
    steps=2,
    memory=MemUpMemoryImpl(embed, mem_transformer),
    loss=PredictorLossStateOnly(predictor, [
        LossModule(nn.MSELoss(), "MSE", 1.0),
    ]),
    data_filter=SeqDataFilterImpl2(rollout),
    info_update=[
        IncrementStep()
    ]
)


memup_iter_eval = MemoryRollout[DataType](
    steps=1000,
    memory=MemUpMemoryImpl(embed, mem_transformer),
    data_filter=SeqDataFilterImpl2(rollout),
    info_update=[
        IncrementStep()
    ]
)

eval_loss = EvalLossStateOnly(predictor, [MSEMetric()])


@torch.no_grad()
def eval(i):

    print("evaluate")
    mem_transformer.eval()
    predictor.eval()

    for x, y, m in test_loader:
        state2 = torch.zeros(x.shape[0], state_length, 128).cuda()

        collector, last_state, _, _ = memup_iter_eval.forward(DataType(x, y, m, x.shape[1]), state2, {}, DataCollectorLastState())
        info2 = {}
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

            loss, state, info, done = memup_iter.forward(DataType(x, y, m, T), state, info, DataCollectorTrainFromState())
            last_info = info['losses']

            loss.backward()
            opt.step()

        print(last_info)
        for name, val in last_info.items():
            writer.add_scalar(f"train/{name}", val, i)

