from copy import deepcopy
import time
from typing import Dict, Tuple
from torch import Tensor, nn
import torch
from transformers.modeling_outputs import TokenClassifierOutput
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel, BertEncoder
from common_modules.pos_encoding import PositionalEncoding2
from common_modules.transformers import DecoderFromBert
from data_filters.sliding_window import SlidingWindowFilter, SlidingWindowWithPadding
from memup.base import SD, DataCollectorAppend, Done, Info, MemUpMemory, SeqDataFilter, State
from torch.nn.utils.rnn import pad_sequence

from memup.loss import TOS


class DataFilter(SeqDataFilter[Dict[str, Tensor]]):

    def __init__(self, center_step: int, context_step: int):
        super().__init__()
        self.center_step = center_step
        self.context_step = context_step
        pos_encoder = PositionalEncoding2(768 * 2, 0, 896)
        self.positions = pos_encoder.forward(torch.zeros(1, 896, 768 * 2)).cuda()

    @torch.no_grad()
    def forward(self, data: Dict[str, Tensor], state: State, info: Info, *args) -> Tuple[SD, Done]:

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
        if stage == "center" and step * BS + BS >= T:
            info["stage"] = "right"
            info["step"] = -1
            print("stage", info["stage"])

        i1 = step * BS
        i2 = i1 + BS

        # print(step, i1, i2)
    
        if stage == "center":
            return self.filter_center(data["center"], i1, i2), False
        else:
            done = (step * BS + BS >= T) and (info["stage"] == "right")
            return self.filter_context(data[stage], i1, i2), done
    

    def filter_center(self, data: Dict[str, Tensor], i1: int, i2: int) -> Dict[str, Tensor]:

        i2 = min(896, i2)
        
        feature_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'bins_mask']
        pad_token_ids = {'input_ids': 3, 'token_type_ids': 0, 'attention_mask': 0, 'bins_mask': 0}
        new_data = {} 

        cusum = data['bins_mask'].type(torch.int32).cumsum(1)
    
        for k in feature_keys:
            new_data[k] = []
            for i in range(cusum.shape[0]):
                mask = (cusum[i] > i1 * 2) * (cusum[i] < i2 * 2) + (cusum[i] == i2 * 2) * data['bins_mask'][i] + (cusum[i] == i1 * 2) * (data['bins_mask'][i] == False) 
                new_data[k].append(data[k][i][mask])

            new_data[k] = pad_sequence(new_data[k], batch_first=True, padding_value=pad_token_ids[k]).cuda()[:, :450]

        new_data["labels"] = data["labels"][:, i1: i2]
        new_data["labels_mask"] = torch.ones(new_data["labels"].shape[:2]).type(torch.bool)
        new_data["positions"] = self.positions[:, i1: i2].expand(new_data["labels"].shape[0], new_data["labels"].shape[1], 768 * 2)
        # print(i1, i2, new_data["input_ids"].shape)

        return new_data
    
    def filter_context(self, data: Dict[str, Tensor], i1: int, i2: int) -> Dict[str, Tensor]:
        
        feature_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        new_data = {} 
    
        for k in feature_keys:
            new_data[k] = data[k][:, i1: i2].cuda()

        return new_data
    


class BertForEnformer(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

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
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        bins_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        positions=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        h = outputs[0]

        hs = torch.cat([h, state], dim=1)
        hs = self.encoder(hs)['last_hidden_state']
        new_state = hs[:, h.shape[1]:]
        out = hs[:, : h.shape[1]]


        return out, h, bins_mask, new_state
    


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, mem_tr: BertForEnformer):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: Dict[str, Tensor], state: State) -> Tuple[Tensor, State]:
        out, hidden, bins_mask, new_state = self.mem_tr.forward(state, **data)

        if data["input_ids"].shape[1] < 450 and bins_mask is not None:
            bins_count = bins_mask[0].type(torch.int32).sum()
            B, D = out.shape[0], out.shape[-1]
            bins_output = torch.cat([
                data["positions"],
                hidden[bins_mask].reshape(B, bins_count // 2, D * 2),
                out[bins_mask].reshape(B, bins_count // 2, D * 2)
            ], dim=-1)
            return bins_output, new_state
        else:
            return None, new_state
    

class DataCollectorTrain(DataCollectorAppend[Dict[str, Tensor], TOS]):
    def apply(self, data: Dict[str, Tensor], out: TokenClassifierOutput, state: State) -> TOS:
        return TOS(data["labels"] if "labels" in data else None, out, state)
    

class Predictor(nn.Module):

    def __init__(self, bert_config):
        super().__init__()
        config2 = deepcopy(bert_config)
        config2.num_attention_heads = 4
        config2.num_hidden_layers = 2
        config2.intermediate_size = bert_config.hidden_size * 2

        self.encoder = BertEncoder(config2)

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size * 2, bert_config.hidden_size * 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size * 2, 5313),
            nn.Softplus()
        )


    def forward(self, x, state):
        B, D = state.shape[0], state.shape[2]
        T = x.shape[1]
        xs = torch.cat([x.reshape(B, T * 6, D), state], dim=1)
        out = self.encoder.forward(xs)['last_hidden_state'][:, 1:T*6:3].reshape(B, T, D * 2)
        return self.head(out)