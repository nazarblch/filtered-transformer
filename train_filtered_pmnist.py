import time
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from filter_model.base import FilteredRecurrentTransformer, NStepFilterObject, FilterModel
from filter_model.chunk_filter import ChunkFilter
from metrics.accuracy import AccuracyMetric
from datasets.pmnist import PermMNISTTaskGenerator
from models.pos_encoding import LinearEmbedWithPos
from models.transformers import RecurrentTransformer, TransformerClassifier

tr_dim = 256
filter_model: FilterModel = NStepFilterObject(20)(ChunkFilter(
    transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(tr_dim, 4, 512, batch_first=True), 3),
    chunk_size=14,
    hidden_dim=tr_dim,
    n_chunks=4
)).cuda()

rec_transformer = FilteredRecurrentTransformer(
    RecurrentTransformer(tr_dim, 4, 4, 512),
    filter_model,
    embedding=LinearEmbedWithPos(1, tr_dim, multiplier=14),
    rollout=2
).cuda()

predictor = TransformerClassifier(10, tr_dim, 4, 2, 512).cuda()

gen = PermMNISTTaskGenerator(is_train=True, padding=False, do_perm=True)
test_gen = PermMNISTTaskGenerator(is_train=False, padding=False, do_perm=True)

writer = SummaryWriter(f"/home/nazar/pomoika/pmnist_{time.time()}")

opt = torch.optim.Adam(
    [{'params': rec_transformer.parameters(), 'lr': 4e-5},
     {'params': predictor.parameters(), 'lr': 4e-5}
     ]
)
scheduler = StepLR(opt, step_size=200, gamma=0.95)

def make_batch(generator, batch_size):
    B = batch_size
    batch = [generator.gen_trajectory() for _ in range(B)]
    X = torch.from_numpy(np.stack([b["x"] for b in batch])).view(B, 28 * 28, 1).cuda()
    Y = torch.from_numpy(np.stack([b["s"] for b in batch])).view(B).cuda()
    return X, Y


for i in range(30000):
    print("iter", i)
    rec_transformer.train()
    predictor.train()
    B = 128
    X, Y = make_batch(gen, B)
    s0 = torch.zeros(B, 30, tr_dim).cuda()
    t0 = time.time()

    states_generator = rec_transformer.forward(X, s0)

    for s in states_generator:
        opt.zero_grad()
        pred = predictor(s)
        loss = nn.CrossEntropyLoss()(pred, Y)
        loss.backward(retain_graph=True)
        opt.step()

    scheduler.step()

    print("iter time", time.time() - t0)
    print("train loss", loss.item())
    writer.add_scalar("train loss", loss.item(), i)

    if i % 20 == 0:
        rec_transformer.eval()
        predictor.eval()
        with torch.no_grad():
            B = 256
            X, Y = make_batch(test_gen, B)
            s0 = torch.zeros(B, 30, tr_dim).cuda()
            *_, last_state = rec_transformer.forward(X, s0)
            pred = predictor(last_state)
            loss = nn.CrossEntropyLoss()(pred, Y)
            acc = AccuracyMetric()(pred, Y)
            print("loss:", loss.item(), "acc:", acc)
            writer.add_scalar("test loss", loss.item(), i)
            writer.add_scalar("test acc", acc, i)

