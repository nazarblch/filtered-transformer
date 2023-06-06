from collections import namedtuple
from copy import deepcopy
import time
from tkinter.messagebox import NO
from types import new_class
from typing import Dict, Optional, Tuple
from typing_extensions import override
from common_modules.rmt import RecurrentTransformerWithStateEmbedding
from torch import Tensor, nn
import torch
from transformers.modeling_outputs import TokenClassifierOutput
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel, BertEncoder
from common_modules.pos_encoding import PositionalEncoding2
from common_modules.transformers import DecoderFromBert
from data_filters.sliding_window import SlidingWindowFilter, SlidingWindowWithPadding
from memup.base import SD, DataCollectorAppend, Done, Info, MemUpMemory, SeqDataFilter, State
from torch.nn.utils.rnn import pad_sequence

from memup.loss import TOS, TOSM


class DataFilter(SeqDataFilter[Dict[str, Tensor]]):

    def __init__(self, step: int):
        super().__init__()
        self.center_step = step
        self.context_step = step
        pos_encoder = PositionalEncoding2(768, 0, 896)
        self.positions = pos_encoder.forward(torch.zeros(1, 896, 768)).cuda()

    @torch.no_grad()
    def forward(self, data: Dict[str, Dict[str, Tensor]], state: State, info: Info, *args) -> Tuple[Dict[str, Tensor], Done]:

        if "stage" not in info:
            info["stage"] = "left"
            print("stage", info["stage"])

        stage = info["stage"]

        BS = self.center_step if stage == "center" else self.context_step
        T = 896 if stage == "center" else data[stage]['input_ids'].shape[1]
        assert "step" in info
        step = info["step"]
    
        if stage == "left" and step * BS + BS >= T:
            info["stage"] = "center"
            info["step"] = -1
            print("stage", info["stage"])
            info["batch_step"] = torch.zeros(data["center"]['input_ids'].shape[0], dtype=torch.int32)
        
        i1 = step * BS
        i2 = i1 + BS
    
        if stage == "center":
            new_data = self.filter_center(data["center"], info)
            
            if info["batch_step"].min() >= T:
                info["stage"] = "right"
                info["step"] = -1
                print("stage", info["stage"])

            return new_data, False
        else:
            done = (step * BS + BS >= T) and (info["stage"] == "right")
            return self.filter_context(data[stage], i1, i2, stage), done
    

    def filter_center(self, data: Dict[str, Tensor], info) -> Dict[str, Tensor]:

        feature_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask']
        pad_token_ids = {'input_ids': 3, 'token_type_ids': 0, 'attention_mask': 0, 'bins_mask': 0, "labels": 0, "labels_mask": 0, "positions": 0, "indices": False}
        new_data = {} 

        cusum = data['bins_mask'].type(torch.int32).cumsum(1)

        for k in feature_keys + [ "labels", "labels_mask", "positions", "indices"]:
            new_data[k] = []
    
        for i in range(cusum.shape[0]):
            i1 = info["batch_step"][i].item()
            i2 =  min(896, i1 + 3)
            mask = (cusum[i] > i1) * (cusum[i] < i2) + (cusum[i] == i2) * data['bins_mask'][i] * (i1 < i2) + (cusum[i] == i1) * (data['bins_mask'][i] == False) * (i1 < i2)
            mask_bk = mask

            assert mask.type(torch.int32).sum() < self.center_step

            while i2 < 896 and mask.type(torch.int32).sum() < self.center_step:
                i2 += 1
                mask_bk = mask
                mask = (cusum[i] > i1) * (cusum[i] < i2) + (cusum[i] == i2) * data['bins_mask'][i] + (cusum[i] == i1) * (data['bins_mask'][i] == False) 
                
            if mask.type(torch.int32).sum() > self.center_step:
                mask = mask_bk
                i2 = i2 - 1

            info["batch_step"][i] = i2

            if data['bins_mask'][i][mask].type(torch.int32).sum().item() != (i2 - i1):
                print(i1, i2, (i2 - i1), data['bins_mask'][i][mask].type(torch.int32).sum().item(), mask.type(torch.int32).sum())

            assert data['bins_mask'][i][mask].type(torch.int32).sum().item() == (i2 - i1)

            for k in feature_keys:
                new_data[k].append(data[k][i][mask])

            labels = data["labels"][i, i1: i2]
            new_data["labels"].append(labels)
            new_data["labels_mask"].append(torch.ones(labels.shape[0]).type(torch.bool))
            new_data["positions"].append(self.positions[0, i1: i2])
            ind_mask = torch.zeros(896).type(torch.bool)
            ind_mask[i1: i2] = True
            new_data["indices"].append(ind_mask)

        for k in feature_keys + [ "labels", "labels_mask", "positions", "indices"]:
            new_data[k] = pad_sequence(new_data[k], batch_first=True, padding_value=pad_token_ids[k]).cuda()

        B = new_data["indices"].shape[0]
        ind_pad = torch.zeros(B, 80).type(torch.bool).cuda()
        new_data["indices"] = torch.cat([ind_pad, new_data["indices"], ind_pad], 1)

        # print(new_data["labels"].shape, new_data["input_ids"].shape)
        new_data["stage"] = "center"

        # print("center block len", new_data["input_ids"].shape[1])

        return new_data
    
    def filter_context(self, data: Dict[str, Tensor], i1: int, i2: int, stage) -> Dict[str, Tensor]:
        
        feature_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask']
        new_data = {} 
    
        for k in feature_keys:
            new_data[k] = data[k][:, i1: i2].cuda()

        lens = new_data['bins_mask'].type(torch.int32).sum(1)
        new_data["labels_mask"] = pad_sequence([
            torch.zeros(l).type(torch.bool) for l in lens
        ], batch_first=True, padding_value=False).cuda() 

        new_data["stage"] = stage

        # print(data['bins_mask'].type(torch.int32).sum(1))
        new_data["indices"] = torch.zeros_like(data['bins_mask']).type(torch.bool)
        new_data["indices"][:, i1: i2][new_data['bins_mask']] = True
        B = new_data["indices"].shape[0]
        new_data["indices"] = new_data["indices"][data['bins_mask']].reshape(B, -1).cuda()

        ind_pad = torch.zeros(B, 896 + 80).type(torch.bool).cuda()
        if stage == "right":
            new_data["indices"] = torch.cat([ind_pad, new_data["indices"]], 1)
        else:
            new_data["indices"] = torch.cat([new_data["indices"], ind_pad], 1)

        assert new_data["indices"].shape[1] == 896 + 80 * 2

        # print(stage, "block len", new_data["input_ids"].shape[1])

        return new_data
    

class BertForEnformer(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.tokenizer = tokenizer

        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.train()

        config2 = deepcopy(config)
        config2.num_attention_heads = 6
        config2.num_hidden_layers = 4
        config2.intermediate_size = config.hidden_size * 2

        self.encoder = BertEncoder(config2)
        self.encoder.train()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        state,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor
    ):
       
        return_dict = self.config.use_return_dict

        prefix = torch.tensor([self.tokenizer.cls_token_id]).cuda()[None, :].repeat(input_ids.shape[0], 1) 
        input_ids = torch.cat([prefix, input_ids], 1)
        attention_mask = torch.cat([torch.ones_like(prefix), attention_mask], dim=1)
        token_type_ids = torch.cat([torch.zeros_like(prefix), token_type_ids], dim=1)        

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        h = outputs[0][:, 1:]

        hs = torch.cat([h, state], dim=1)
        hs = self.encoder(hs)['last_hidden_state']
        new_state = hs[:, h.shape[1]:]
        out = hs[:, : h.shape[1]]

        empty_mask = attention_mask[:, 1:].type(torch.int32).sum(1)
        new_state[empty_mask == 0] = state[empty_mask == 0]

        return out, h, new_state
    

class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, mem_tr: BertForEnformer):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: Dict[str, Tensor], state: State) -> Tuple[Tensor, State]:
        out, hidden, new_state = self.mem_tr.forward(state, data["input_ids"], data["attention_mask"], data["token_type_ids"])
        bins_mask = data['bins_mask'] 

        lens = bins_mask.type(torch.int32).sum(1)
        lens = lens.cpu().type(torch.int32).numpy().tolist()
        B, D = out.shape[0], out.shape[-1]
        bins_output = torch.cat([
            # data["positions"],
            pad_sequence(torch.split(hidden[bins_mask].reshape(-1, D), lens), batch_first=True) +
            pad_sequence(torch.split(out[bins_mask].reshape(-1, D), lens), batch_first=True)
        ], dim=-1)
        return bins_output, new_state
       
        

class MemUpMemoryRMT(MemUpMemory):

    def __init__(self, mem_tr: RecurrentTransformerWithStateEmbedding):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: Dict[str, Tensor], state: State) -> Tuple[Tensor, State]:
        bert_out = self.mem_tr.forward(data, state)
        bins_mask = data["bins_mask"] 
        out, new_state = bert_out.out, bert_out.state

        lens = bins_mask.type(torch.int32).sum(1)
        lens = lens.cpu().type(torch.int32).numpy().tolist()
        B, D = out.shape[0], out.shape[-1]
        bins_output = torch.cat([
            # data["positions"],
            pad_sequence(torch.split(out[bins_mask].reshape(-1, D), lens), batch_first=True)
        ], dim=-1)
        return bins_output, new_state
    

TrainBatch = namedtuple("TrainBatch", ["out", "state", "mask", "global_mask"])

class DataCollectorTrain(DataCollectorAppend[Dict[str, Tensor], TrainBatch]):
    def apply(self, data: Dict[str, Tensor], out: TokenClassifierOutput, state: State) -> TrainBatch:
        mask = data["labels_mask"] 
        global_mask = data["indices"] 

        return TrainBatch(out, state, mask, global_mask)
    

class ContextCollector(DataCollectorAppend[Dict[str, Tensor], Tensor]):
    def apply(self, data:  Dict[str, Tensor], out: Tensor, state: State) -> Optional[Tuple[Tensor, Tensor]]:
        return (out.cpu(), data["labels_mask"].cpu()) 
    
    @override
    def result(self, cat_dims: Tuple[int] = ..., cat_keys: Tuple[str] = ...):
        context = torch.cat([c for c, _ in self.collection], 1)
        c_mask = torch.cat([m for _, m in self.collection], 1)
        B, _, D = context.shape
        context = context[c_mask].reshape(B, -1, D)
        return context
    

class LinearPredictor(nn.Module):

    def __init__(self, bert_config, mult=1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(bert_config.hidden_size * mult, bert_config.hidden_size * mult),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size * mult, 5313),
            nn.Softplus()
        )

    def forward(self, x):

        return self.head(x)

    
    

class Predictor(nn.Module):

    def __init__(self, bert_config, mult=1):
        super().__init__()
        config2 = deepcopy(bert_config)
        config2.hidden_size = bert_config.hidden_size * mult
        config2.num_attention_heads = 4
        config2.num_hidden_layers = 4
        # config2.intermediate_size = bert_config.hidden_size * mult * 2

        self.encoder = BertEncoder(config2)
        self.config = config2
        self.mult = mult

        self.head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(bert_config.hidden_size * mult, bert_config.hidden_size * mult),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size * mult, 5313),
            nn.Softplus()
        )

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int]
    ) -> Tensor:
        
        dtype = torch.float32

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask



    def forward(self, x, state):
        B, D = state.shape[0], state.shape[2]
        T = x.shape[1]
        # mult = x.shape[2] // D
        # extended_mask = mask[:, :, None].expand(*mask.shape, mult).reshape(B, T * mult).type(torch.int32)
        # extended_mask = mask
        # state_3 = torch.cat([state] * self.mult, -1)
        # state_mask = torch.ones(state.shape[:2], dtype=torch.int32, device=state.device)
        # extended_mask = torch.cat([extended_mask, state_mask], dim=1)
        xs = torch.cat([x, state], dim=1)
        # xs = x
        # extended_mask = self.get_extended_attention_mask(extended_mask, xs.shape)
        out = self.encoder.forward(xs)['last_hidden_state'][:, :T][:, 80:-80]
        assert out.shape[1] == 896
        return self.head(out)