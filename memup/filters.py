from typing import Generic, Dict, Any
from memup.base import SeqDataFilter, SD, MemUpMemory, State
from models.transformers import RecurrentTransformer
import torch


class ContextPreprocessor(Generic[SD]):
    def __init__(self, mem_transformer: MemUpMemory[SD], key="context"):
        self.mem_transformer = mem_transformer

    def forward(self, data: SD, state_0: State, info: Dict[str, Any], seq_filter: SeqDataFilter[SD]) -> Dict[str, Any]:
        state = state_0
        filtered_data = seq_filter.filter_data(state, data)
        out_collection = []

        while filtered_data is not None:
            o, s = self.mem_transformer.forward(filtered_data, state)
            state = os.state
            out.append(os.out[:, : filtered_data.target.shape[1]])
            filtered_data = seq_filter.filter_data(state, data)

        context = torch.cat(out, dim=1)




class SeqDataFilterImpl(SeqDataFilter[SD], Generic[SD]):

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