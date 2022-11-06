import time
from functools import reduce
from itertools import chain
import numpy as np
from hfilter.filtered_transformer import ChunkFilter, FilteredTransformer, HierarchicalChunkFilter, ChunkFilter2, \
    HierarchicalTransformer, SlideFilter, RandomChunkFilter
import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from metrics import AccuracyMetric
from pmnist import PermMNISTTaskGenerator

# torch.cuda.set_device("cuda:0")

class T22(nn.Transformer):
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.encoder(xy)[:, x.shape[1]:]

tr_dim = 256
filter_model = ChunkFilter(nn.Transformer(tr_dim, 4, 3, 3, 512, batch_first=True).encoder, 14, 1, tr_dim, 4).cuda()
# filter_model = RandomChunkFilter(7, 1, tr_dim, 10).cuda()
# filter_model = HierarchicalChunkFilter(
#         HierarchicalTransformer(
#             T22(tr_dim, 4, 3, 3, 512, batch_first=True),
#             T22(tr_dim, 4, 1, 1, 512, batch_first=True),
#             T22(tr_dim, 4, 1, 1, 512, batch_first=True),
#             T22(tr_dim, 4, 1, 1, 512, batch_first=True),
#             dim=1, chunk_size=7),
#         sample_size=10
#     )
f_tr = FilteredTransformer(T22(tr_dim, 4, 4, 4, 512, batch_first=True), filter_model, 2).cuda()

predictor_tr = nn.Transformer(tr_dim, 4, 2, 2, 512, batch_first=True).encoder.cuda()
predictor = nn.Linear(tr_dim, 10).cuda()

gen = PermMNISTTaskGenerator(is_train=True, padding=False, do_perm=True)
test_gen = PermMNISTTaskGenerator(is_train=False, padding=False, do_perm=True)

writer = SummaryWriter(f"/home/nazar/pomoika/pmnist_{time.time()}")

opt = torch.optim.Adam(
    [{'params': f_tr.parameters(), 'lr': 1e-5},
     {'params': predictor_tr.parameters(), 'lr': 1e-5},
     {'params': predictor.parameters(), 'lr': 1e-5}
     ]
)

def make_batch(generator, batch_size):
    B = batch_size
    batch = [generator.gen_trajectory() for _ in range(B)]
    X = torch.from_numpy(np.stack([b["x"] for b in batch])).view(B, 28 * 28, 1).cuda()
    Y = torch.from_numpy(np.stack([b["s"] for b in batch])).view(B).cuda()
    return X, Y


for i in range(30000):
    print("iter", i)
    f_tr.train()
    predictor_tr.train()
    B = 128
    X, Y = make_batch(gen, B)
    s = torch.zeros(B, 30, tr_dim).cuda()
    t0 = time.time()

    for k in range(15):
        opt.zero_grad()
        s = f_tr.forward(s, X)
        pred = predictor_tr(s)
        pred = predictor(pred[:, -1])
        loss = nn.CrossEntropyLoss()(pred, Y)
        loss.backward()
        opt.step()
        s = s.detach()

    print("iter time", time.time() - t0)
    print("train loss", loss.item())
    writer.add_scalar("train loss", loss.item(), i)

    if i % 20 == 0:
        f_tr.eval()
        predictor_tr.eval()
        with torch.no_grad():
            B = 256
            X, Y = make_batch(test_gen, B)
            s = torch.zeros(B, 30, tr_dim).cuda()
            for _ in range(15):
                s = f_tr.forward(s, X)
            pred = predictor_tr(s)
            pred = predictor(pred[:, -1])
            loss = nn.CrossEntropyLoss()(pred, Y)
            acc = AccuracyMetric()(pred, Y)
            print("loss:", loss.item(), "acc:", acc)
            writer.add_scalar("test loss", loss.item(), i)
            writer.add_scalar("test acc", acc, i)

