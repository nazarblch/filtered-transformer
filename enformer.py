from collections import namedtuple
from typing import Iterator, Tuple, List, Callable, Optional
import torch
from gena_lm.modeling_bert import BertModel, BertForSequenceClassification
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets.enformer_h5 import EnformerDataset
from memup.base import SeqDataFilter, MemUpMemory, MemUpLoss, MemUpLossIterator, State
from models.transformers import BertRecurrentTransformer, RecurrentTransformerFromBert, \
    BertRecurrentTransformerWithTokenizer

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
train_data = EnformerDataset(["/home/nazar/PycharmProjects/enformer/train_0.h5"])

train_loader = DataLoader(train_data, shuffle=True, batch_size=16)

bert: BertModel = BertModel.from_pretrained('AIRI-Institute/gena-lm-bert-base')
mem_transformer = BertRecurrentTransformerWithTokenizer(bert, tokenizer, 300, 12, 4, bert.config.hidden_size * 2).cuda()


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = RecurrentTransformerFromBert(bert, 12, 2, bert.config.hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(bert.config.hidden_size, 5313)
        )

    def forward(self, x, state):
        out = self.encoder.forward(x, state).out
        return self.head(out)


predictor = Predictor().cuda()

opt = torch.optim.Adam([
    {"params": mem_transformer.bert.parameters(), "lr": 4e-6},
    {"params": mem_transformer.encoder.parameters(), "lr": 2e-5},
    {"params": predictor.parameters(), "lr": 2e-5}
])


device = torch.device("cuda")
BS = 1280
TOPK = 10

@torch.no_grad()
def sample_extra_train_data(count: int, context: Tensor, errors: Tensor, target: Tensor):
    probs = errors / errors.sum(dim=1, keepdim=True)
    sample = torch.multinomial(probs, count, replacement=False)
    index = sample[:, :, None]
    B = context.shape[0]
    return torch.gather(context, 1, index.expand(-1, -1, context.shape[-1])).reshape(B, count, context.shape[-1]), \
           torch.gather(target, 1, index.expand(-1, -1, target.shape[-1])).reshape(B, count, target.shape[-1])


DataType = namedtuple("DataType", ["text", "target", "coords"])
DataTypeWithMemory = Tuple[DataType, Tensor, Tensor]


class SeqDataFilterImpl(SeqDataFilter[DataType]):

    def __init__(self):
        super().__init__()
        self.m = 0
        self.sd = 1

    @torch.no_grad()
    def preproc(self, data: DataType):
        state: Tensor = torch.zeros(data.target.shape[0], 50, bert.config.hidden_size, device=device)
        out = []
        T = len(data.text[0])
        step = 0

        while step * BS < T:
            filtered_data = self.filter_data(state, data, step, T)
            os = mem_transformer.forward(filtered_data.text, state)
            state = os.state
            out.append(os.out[:, : filtered_data.target.shape[1]])
            step += 1

        context = torch.cat(out, dim=1)
        pred = predictor(context, state)
        self.m = self.m * 0.99 + torch.mean(data.target, dim=0, keepdim=True) * 0.01
        self.sd = self.sd * 0.99 + torch.std(data.target, dim=0, keepdim=True) * 0.01
        m, sd = self.m, self.sd
        errors = nn.MSELoss(reduction='none')((pred.cpu() - m)/sd, (data.target - m)/sd).mean(dim=2)
        errors1 = nn.MSELoss(reduction='none')(pred.cpu(), data.target).mean(dim=2)
        print("err", errors1.mean().item())

        return context.cpu(), errors.cpu()

    def extend_data(self, context, errors, target) -> Tuple[Tensor, Tensor]:

        o, t = sample_extra_train_data(TOPK, context, errors, target)
        return o, t

    def filter_data(self, state: Tensor, data, step: int, max_len: int) -> DataType:

        i1 = step * BS
        i2 = i1 + BS
        i1_pad = max(0, i1 - BS // 10)
        i2_pad = min(max_len, i2 + BS // 10)

        pad_text = [t[i1_pad:i2_pad] for t in data.text]
        filtered_target = data.target[(data.coords >= i1) * (i2 > data.coords)].view(data.target.shape[0], -1,
                                                                                     data.target.shape[2])
        filtered_coords = data.coords[(data.coords >= i1) * (i2 > data.coords)].view(data.coords.shape[0], -1)

        return DataType(pad_text, filtered_target, filtered_coords)

    def forward(self, data: DataType, state: State) -> Tuple[Optional[DataType], State]:
        T = len(data.text[0])
        step = state.extra["step"] if "step" in state.extra else 0

        if step * BS >= T:
            return None, state

        if step % 20 == 0:
            state.extra["context"], state.extra["err"] = self.preproc(data)

        filtered_data = self.filter_data(state.state, data, step, T)
        if (step + 1) % 2 == 0:
            context, context_target = self.extend_data(state.extra["context"], state.extra["err"], data.target)
            state.extra["context_selected"], state.extra["context_target"] = context, context_target

        state.extra["step"] = step + 1

        return filtered_data, state


class MemUpMemoryImpl(MemUpMemory):
    def forward(self, data: DataType, state: Tensor) -> Tuple[Tensor, Tensor]:
        os = mem_transformer.forward(data.text, state)
        return os.out[:, :data.target.shape[1]], os.state


class MemUpLossImpl(MemUpLoss):
    def forward(self, data: List[DataTypeWithMemory], state: State) -> Tensor:
        out, target = torch.cat([d[1] for d in data], 1), torch.cat([d[0].target for d in data], 1)
        s0 = torch.cat([d[2] for d in data], 0)

        context = state.extra["context_selected"].cuda()
        context_target = state.extra["context_target"]

        out = torch.cat([out, context], 1)
        out = torch.cat([out]*len(data), 0)
        target = torch.cat([target, context_target], 1)
        target = torch.cat([target] * len(data), 0)

        pred = predictor(out, s0)
        loss = nn.MSELoss()(pred, target.cuda())

        return loss


memup_iter = MemUpLossIterator[DataType](
    rollout=2,
    memory=MemUpMemoryImpl(),
    loss=MemUpLossImpl(),
    data_filter=SeqDataFilterImpl(),
)


for text1, target1, coords1 in train_loader:

    state = State(torch.zeros(target1.shape[0], 50, bert.config.hidden_size, device=device), {})
    done = False

    while not done:
        opt.zero_grad()
        loss, state, done = memup_iter.forward(DataType(text1, target1, coords1), state)
        print(loss.item())
        if loss is not None:
            loss.backward()
        opt.step()
