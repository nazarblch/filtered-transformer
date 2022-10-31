import time
from functools import reduce
from itertools import chain
import numpy as np
from hfilter.filtered_transformer import ChunkFilter, FilteredTransformer, HierarchicalChunkFilter, ChunkFilter2, \
    HierarchicalTransformer, RandomChunkFilter, SlideFilter
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, Tensor

from location_gen import LocationDetect, AddTask
from metrics import AccuracyMetric
from pmnist import PermMNISTTaskGenerator

torch.cuda.set_device("cuda:0")

class T22(nn.LSTM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, hc):
        h0, c0 = hc
        output, (hn, cn) = super().forward(input, (h0, c0))
        return hn, cn

tr_dim = 256
filter_model = SlideFilter(25, 2, tr_dim).cuda()
# filter_model = HierarchicalChunkFilter(
#         HierarchicalTransformer(
#             nn.Transformer(tr_dim, 4, 3, 3, 512, batch_first=True),
#             nn.Transformer(tr_dim, 4, 1, 1, 512, batch_first=True),
#             nn.Transformer(tr_dim, 4, 1, 1, 512, batch_first=True),
#             nn.Transformer(tr_dim, 4, 1, 1, 512, batch_first=True),
#             dim=2, chunk_size=10),
#         sample_size=10
#     )
f_tr = FilteredTransformer(T22(tr_dim, tr_dim, 2, batch_first=True, dropout=0), filter_model, 2).cuda()

predictor = nn.Sequential(
    nn.Linear(tr_dim, tr_dim),
    nn.ReLU(inplace=True),
    nn.Linear(tr_dim, tr_dim),
    nn.ReLU(inplace=True),
    nn.Linear(tr_dim, tr_dim)
).cuda()

gen = AddTask(500)
test_gen = AddTask(500)

writer = SummaryWriter(f"/home/jovyan/pomoika/add_random_{time.time()}")

opt = torch.optim.Adam(
    [{'params': f_tr.parameters(), 'lr': 1e-5},
     {'params': predictor.parameters(), 'lr': 1e-5}
     ]
)

def make_batch(generator, batch_size):
    B = batch_size
    X, Y = generator.gen_batch(B)
    X, Y = X.cuda(), Y.cuda()
    return X, Y


for i in range(10000):
    # print("iter", i)
    f_tr.train()
    B = 128
    X, Y = make_batch(gen, B)

    h = torch.randn(2, B, tr_dim).cuda()
    c = torch.randn(2, B, tr_dim).cuda()

    for k in range(10):
        opt.zero_grad()
        h, c = f_tr.forward((h, c), X)
        pred = predictor(h[-1, :, :])
        loss = nn.MSELoss()(pred, Y)
        loss.backward()
        opt.step()
        h, c = h.detach(), c.detach()
    print(i, loss.item())
    writer.add_scalar("train mse", loss.item(), i)

    if i % 10 == 0:
        f_tr.eval()
        with torch.no_grad():
            B = 128
            X, Y = make_batch(gen, B)
            h = torch.randn(2, B, tr_dim).cuda()
            c = torch.randn(2, B, tr_dim).cuda()

            for k in range(10):
                h, c = f_tr.forward((h, c), X)
            pred = predictor(h[-1, :, :])
            loss = nn.MSELoss()(pred, Y)
            print("test", loss.item())
            writer.add_scalar("test mse", loss.item(), i)


