from collections import namedtuple
from copy import deepcopy
import time
from tkinter.messagebox import NO
from types import new_class
from typing import Dict, Optional, Tuple
from typing_extensions import override

from sympy import re
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
        # pos_encoder = PositionalEncoding2(768, 0, 896)
        # self.positions = pos_encoder.forward(torch.zeros(1, 896, 768)).cuda()

    @torch.no_grad()
    def forward(self, data: Dict[str, Tensor], state: State, info: Info, *args) -> Tuple[Dict[str, Tensor], Done]:

        BS = self.center_step
        T = data['input_ids'].shape[1]
        
        step = info["step"]
        i1 = step * BS
        i2 = i1 + BS
    
        done = (i2 >= T)
        
        return self.filter_context(data, i1, i2), done
    

    def filter_context(self, data: Dict[str, Tensor], i1: int, i2: int) -> Dict[str, Tensor]:
        
        feature_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask']
        new_data = {} 
    
        for k in feature_keys:
            new_data[k] = data[k][:, i1: i2].cuda()

        new_data["labels_inject"] = torch.zeros_like(data['bins_mask']).type(torch.bool)
        new_data["labels_inject"][:, i1: i2][new_data['bins_mask']] = True
        new_data["labels_inject"] = new_data["labels_inject"][data['bins_mask']].cuda()

        lens = data['bins_mask'].type(torch.int32).sum(1)
        lens = lens.cpu().type(torch.int32).numpy().tolist()
        new_data["labels_inject"] = pad_sequence(torch.split(new_data["labels_inject"], lens), batch_first=True, padding_value=False)

        # print(i1, i2, new_data["labels_inject"][0])

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
        config2.num_hidden_layers = 6
        config2.intermediate_size = config.hidden_size * 2

        self.encoder = BertEncoder(config2)
        self.encoder.train()
        self.post_init()

    def forward(
        self,
        state: Tensor,
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

        bins_output = hidden[bins_mask] + out[bins_mask]
        
        return bins_output, new_state


class ContextCollector(DataCollectorAppend[Dict[str, Tensor], Tensor]):
    def apply(self, data:  Dict[str, Tensor], out: Tensor, state: State) -> Optional[Tuple[Tensor, Tensor]]:
        return (out.cpu(), data["labels_inject"].cpu()) 
    
    @override
    def result(self, cat_dims: Tuple[int] = ..., cat_keys: Tuple[str] = ...):
        m0 = self.collection[0][1]
        context = torch.zeros(m0.shape[0], m0.shape[1], 768, device=m0.device)
        for c, m in self.collection:
            context[m] = c
            m0 = m0 + m

        return context
    

TrainBatch = namedtuple("TrainBatch", ["out", "state", "mask"])

class DataCollectorTrain(DataCollectorAppend[Dict[str, Tensor], TrainBatch]):
    def apply(self, data: Dict[str, Tensor], out: Tensor, state: State) -> TrainBatch:
        return TrainBatch(out, state, data["labels_inject"]) 