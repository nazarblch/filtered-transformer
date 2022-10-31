from mbrl.Trainer import ContextChangeModel, StochasticBinaryTensor
from torch import Tensor, nn
import torch
from models import FloatTransformer


class Change(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            FloatTransformer(1, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, state: Tensor) -> StochasticBinaryTensor:
        return StochasticBinaryTensor(self.net(state))


x = []
y = []

for i in range(1000):
    x.append(torch.randn(64, 1) + i % 10)
    yi = torch.zeros(64, 1)
    yi[torch.arange(64) % 10 == i % 10] = 1
    y.append(yi)

x = torch.stack(x)
y = torch.stack(y)

model = Change().cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

B = 50
N = x.shape[0]
for i in range(10000):
    b1 = (i * B) % N
    b2 = (i * B + B) % N
    if b1 >= b2 or b1 > N - B:
        continue

    opt.zero_grad()
    batch = x[b1:b2].cuda()
    target = y[b1:b2].cuda()

    pred = model.forward(batch).sample()[0]
    loss = nn.MSELoss()(pred, target)
    print(loss.item())
    loss.backward()
    opt.step()
