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

class T22(nn.Transformer):
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.encoder(xy)[:, x.shape[1]:]

tr_dim = 256
filter_model = SlideFilter(50, 2, tr_dim).cuda()
# filter_model = HierarchicalChunkFilter(
#         HierarchicalTransformer(
#             nn.Transformer(tr_dim, 4, 3, 3, 512, batch_first=True),
#             nn.Transformer(tr_dim, 4, 1, 1, 512, batch_first=True),
#             nn.Transformer(tr_dim, 4, 1, 1, 512, batch_first=True),
#             nn.Transformer(tr_dim, 4, 1, 1, 512, batch_first=True),
#             dim=2, chunk_size=10),
#         sample_size=10
#     )
f_tr = FilteredTransformer(T22(tr_dim, 4, 2, 2, 512, batch_first=True, dropout=0), filter_model, 2).cuda()

predictor_tr = nn.Transformer(tr_dim, 4, 2, 2, 512, batch_first=True, dropout=0).encoder.cuda()
predictor = nn.Linear(tr_dim, 1).cuda()

gen = AddTask(1000)
test_gen = AddTask(1000)

writer = SummaryWriter(f"/home/jovyan/pomoika/add_random_{time.time()}")

opt = torch.optim.Adam(
    [{'params': f_tr.parameters(), 'lr': 1e-5},
     {'params': predictor_tr.parameters(), 'lr': 1e-5},
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
    predictor_tr.train()
    B = 128
    X, Y = make_batch(gen, B)
    s = torch.zeros(B, 30, tr_dim).cuda()

    for k in range(10):
        opt.zero_grad()
        s = f_tr.forward(s, X)
        pred = predictor_tr(s)
        pred = predictor(pred[:, -1])
        loss = nn.MSELoss()(pred, Y)
        loss.backward()
        opt.step()
        s = s.detach()
    print(i, loss.item())
    writer.add_scalar("train mse", loss.item(), i)

    if i % 10 == 0:
        f_tr.eval()
        predictor_tr.eval()
        with torch.no_grad():
            B = 128
            X, Y = make_batch(gen, B)
            s = torch.zeros(B, 30, tr_dim).cuda()

            for k in range(10):
                s = f_tr.forward(s, X)
            pred = predictor_tr(s)
            pred = predictor(pred[:, -1])
            loss = nn.MSELoss()(pred, Y)
            print("test", loss.item())
            writer.add_scalar("test mse", loss.item(), i)


