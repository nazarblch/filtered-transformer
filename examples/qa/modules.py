import random
from turtle import forward
from typing import Dict, Tuple
from common_modules import pos_encoding
from common_modules.pos_encoding import PositionalEncoding2
from common_modules.rmt import RecurrentTransformerWithStateEmbedding
from memup.base import Done, Info, State
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from transformers import RobertaModel


class DataFilter(nn.Module):

    def __init__(self, tokenizer, size) -> None:
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))
        self.segment_size = size

    def pad_add_special_tokens_for_qa(self, tensor, query_option):
        input_elements = [self.cls_token, tensor, query_option]
        tensor = torch.cat(input_elements)

        pad_size = self.segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
        return tensor
    
    def pad_add_special_tokens_for_context(self, tensor):
        input_elements = [self.cls_token, tensor, self.sep_token]
        tensor = torch.cat(input_elements)

        pad_size = self.segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
        return tensor
    
    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask

    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)
    
    def get_cut_input_without_option(self, input_ids, input_part_token_start_idx, shift_batch):
        # input_ids -> (b_s, 4, 4098)
        B, _, T = input_ids.shape
        input_ids = input_ids.reshape(B * 4, T)
        input_part_token_start_idx = input_part_token_start_idx.reshape(B * 4)

        end_seq = []
        ss_batch = []
    
        for i, seq in enumerate(input_ids):
            # seq = cls + context + sep + query + option + sep
            spliter_inx = input_part_token_start_idx[i]
            context = seq[:spliter_inx]  # cls + context
            context = context[(context != self.pad_token_id) & (context != self.cls_token_id) & (context != self.sep_token_id)]  # context

            start = shift_batch[i].item()
            end = max(start, start + self.segment_size - 2)
            end = min(end, len(context))
            # print(start, end)
            input_segment = context[start:end] 
            input_segment = self.pad_add_special_tokens_for_context(input_segment)
            assert len(input_segment) == self.segment_size
            ss_batch.append(input_segment)
            end_seq.append(end)

        # print("ends", torch.stack(end_seq))
        shift_batch = torch.tensor(end_seq)
        ss_batch = torch.stack(ss_batch)

        return ss_batch, shift_batch


    def get_cut_input(self, input_ids, input_part_token_start_idx, shift_batch):
        # input_ids -> (b_s, 4, 4098)
        B, _, T = input_ids.shape
        input_ids = input_ids.reshape(B * 4, T)
        input_part_token_start_idx = input_part_token_start_idx.reshape(B * 4)

        end_seq = []
        ss_batch = []
    
        for i, seq in enumerate(input_ids):
            # seq = cls + context + sep + query + option + sep
            spliter_inx = input_part_token_start_idx[i]
            query_option = seq[spliter_inx:]
            query_option = query_option[(query_option != self.pad_token_id) & (query_option != self.cls_token_id)]  # sep + query + option + sep
            context = seq[:spliter_inx]  # cls + context
            context = context[(context != self.pad_token_id) & (context != self.cls_token_id) & (context != self.sep_token_id)]  # context

            start = shift_batch[i].item()
            end = max(start, start + self.segment_size - len(query_option) - 1)
            end = min(end, len(context))
            # print(start, end)
            input_segment = context[start:end] 
            input_segment = self.pad_add_special_tokens_for_qa(input_segment, query_option)
            assert len(input_segment) == self.segment_size
            ss_batch.append(input_segment)
            end_seq.append(end)

        # print("ends", torch.stack(end_seq))
        shift_batch = torch.tensor(end_seq)
        ss_batch = torch.stack(ss_batch)

        return ss_batch, shift_batch
    
    def forward(self, batch: Dict[str, Tensor], state: State = None, info: Info = {}, *args) -> Tuple[Dict[str, Tensor], Done]:

        if "shift_batch" not in info:
            info["shift_batch"] = torch.zeros(batch['input_ids'].shape[0] * 4, dtype=torch.int32)
        
        # 
        if random.randint(0, 10) > 5 or info["shift_batch"].sum() < 1:
            input_ids, new_shift = self.get_cut_input(batch['input_ids'], batch['input_part_token_start_idx'], info["shift_batch"])
        else:
            input_ids, new_shift = self.get_cut_input_without_option(batch['input_ids'], batch['input_part_token_start_idx'], info["shift_batch"])
        
        done = (info["shift_batch"] - new_shift).abs().sum().item() == 0
        info["shift_batch"] = new_shift

        return {
            "label": batch["label"].cuda(),
            'input_ids': input_ids.cuda(),
            'attention_mask': self.get_attention_mask(input_ids).cuda(),
            'token_type_ids': self.get_token_type_ids(input_ids).cuda()
        }, done


from copy import deepcopy


class Predictor(nn.Module):

    def __init__(self, bert_config):
        super().__init__()
        config2 = deepcopy(bert_config)
        config2.hidden_size = bert_config.hidden_size 
        config2.num_attention_heads = 2
        config2.num_hidden_layers = 2
        config2.hidden_dropout_prob = 0.1
        config2.intermediate_size = 768

        self.encoder = RobertaModel(config2).encoder
        self.encoder.train()
        self.config = config2

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, 1),
            # nn.Dropout(0.1),
            # nn.ReLU(),
            # nn.Linear(bert_config.hidden_size, 4),
        )

    def forward(self, state):
        B, D = state.shape[0], state.shape[2]
        print(state.shape)
        out = self.encoder.forward(state)['last_hidden_state'][:, -1]
        return self.head(out).reshape(B // 4, 4)
    


from memup.base import MemUpMemory

class RobertaRT(nn.Module):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, roberta: RobertaModel):
        super().__init__()

        self.bert = roberta
        self.bert.config.hidden_dropout_prob = 0.1
        self.bert.train()

        config2 = deepcopy(roberta.config)
        config2.num_attention_heads = 2
        config2.num_hidden_layers = 3
        config2.hidden_dropout_prob = 0.1
        config2.intermediate_size = 768

        self.encoder = RobertaModel(config2).encoder
        self.encoder.train()

        # self.pos_encoder = PositionalEncoding2(768, 0.1, 500)

    def forward(
        self,
        state,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor
    ):
              
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        h = outputs[0]

        hs = torch.cat([h, state], dim=1)
        # hs = self.pos_encoder(hs)
        hs = self.encoder(hs)['last_hidden_state']
        new_state = hs[:, h.shape[1]:]
        # out = hs[:, : h.shape[1]]

        empty_mask = attention_mask.type(torch.int32).sum(1)
        new_state[empty_mask == 0] = state[empty_mask == 0]

        return new_state
    
    
class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, mem_tr: RobertaRT):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: Dict[str, Tensor], state: State) -> Tuple[Tensor, State]:
        new_state = self.mem_tr.forward(state, data["input_ids"], data["attention_mask"], data["token_type_ids"])
        return None, new_state
    


class MemUpMemoryRMT(MemUpMemory):

    def __init__(self, mem_tr: RecurrentTransformerWithStateEmbedding):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: Dict[str, Tensor], state: State) -> Tuple[Tensor, State]:
        data["input_ids"] = data["input_ids"][:, 1:]
        data["attention_mask"] = data["attention_mask"][:, 1:]
        data["token_type_ids"] =  data["token_type_ids"][:, 1:]
        bert_out = self.mem_tr.forward(data, state)
        return None, bert_out.state

    



