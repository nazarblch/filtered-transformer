from copy import deepcopy
from typing import Dict, Optional, Tuple
from typing_extensions import override
from gena_lm.modeling_bert import BertPreTrainedModel, BertModel, BertEncoder
from data_filters.top_errors import InputTargetMask
from memup.loss import TOSM
from torch import Tensor
from memup.base import DataCollectorAppend, MemUpMemory, State
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput


class BertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        if getattr(self.config, 'problem_type', None) is None:
            self.config.problem_type = 'single_label_classification'

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_mask=None,
        pos_weight=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                if labels_mask is None:
                    loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
                    loss = loss_fct(logits, labels)
                else:
                    loss_fct = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
                    loss = loss_fct(logits, labels)
                    loss = loss * labels_mask.unsqueeze(-1)
                    loss = loss.sum() / labels_mask.sum() if labels_mask.sum() != 0.0 else torch.tensor(0.0, device=logits.device)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class BertForSpliceAI(BertPreTrainedModel):

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
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        labels_ohe=None,
        labels_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        positions=None,
        pos_weight=None,
        length=-1
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prefix = torch.tensor([self.tokenizer.cls_token_id]).cuda()[None, :].repeat(input_ids.shape[0], 1) 
        suff = torch.tensor([self.tokenizer.sep_token_id]).cuda()[None, :].repeat(input_ids.shape[0], 1) 

        input_ids = torch.cat([prefix, input_ids, suff], 1)
        attention_mask = torch.cat([torch.ones_like(prefix), attention_mask, torch.ones_like(suff)], dim=1)
        token_type_ids = torch.cat([torch.zeros_like(prefix), token_type_ids, torch.zeros_like(suff)], dim=1)  

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
        h = h[:, 1: h.shape[1]-1]

        hs = torch.cat([h, state], dim=1)
        hs = self.encoder(hs)['last_hidden_state']
        new_state = hs[:, h.shape[1]:]
        out = hs[:, : h.shape[1]]

        empty_mask = attention_mask[:, 1: h.shape[1]-1].type(torch.int32).sum(1)
        new_state[empty_mask == 0] = state[empty_mask == 0]

        return out, h, labels_mask.type(torch.bool), new_state
    


class MemUpMemoryImpl(MemUpMemory):

    def __init__(self, mem_tr: BertForSpliceAI):
        super().__init__()
        self.mem_tr = mem_tr

    def forward(self, data: Dict[str, Tensor], state: State) -> Tuple[Optional[Tensor], State]:
        out, hidden, bins_mask, new_state = self.mem_tr.forward(state, **data)

        lens = bins_mask.type(torch.int32).sum(1)
        lens = lens.cpu().type(torch.int32).numpy().tolist()
        B, D = out.shape[0], out.shape[-1]
        bins_output = torch.cat([
            # hidden + data["positions"],
            out
        ], dim=-1)
        
        if sum(lens) > 0:
            return bins_output, new_state
        else:
            return None, new_state
        

class DataCollectorTrain(DataCollectorAppend[Dict[str, Tensor], TOSM]):
    def apply(self, data: Dict[str, Tensor], out: Tensor, state: State) -> TOSM:

        mask = data["labels_mask"]
        labels = data["labels"]

        return TOSM(labels, out, state, mask) if out is not None else TOSM(None, None, state, None)
    

class ContextCollector(DataCollectorAppend[Dict[str, Tensor], Tuple[Tensor, Tensor, Tensor]]):
    def apply(self, data:  Dict[str, Tensor], out: Tensor, state: State) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        mask = data["labels_mask"]
        labels = data["labels"]
        return (out.cpu(), mask.cpu(), labels.cpu()) if out is not None else None
    
    @override
    def result(self, cat_dims: Tuple[int] = ..., cat_keys: Tuple[str] = ...):
        return [InputTargetMask(o, l, m, o.shape[1]) for o, m, l in self.collection]
    

class Predictor(nn.Module):

    def __init__(self, bert_config):
        super().__init__()
        config2 = deepcopy(bert_config)
        config2.num_attention_heads = 4
        config2.num_hidden_layers = 3
        config2.intermediate_size = bert_config.hidden_size * 2

        self.encoder = BertEncoder(config2)
        self.config = config2

        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size, 3)
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



    def forward(self, x, state, mask):
        B, D = state.shape[0], state.shape[2]
        T = x.shape[1]
        assert mask.shape[1] == T
        print(B, "T=", T, "D=", D)
        mult = x.shape[2] // D
        extended_mask = mask[:, :, None].expand(*mask.shape, mult).reshape(B, T * mult).type(torch.int32)
        state_mask = torch.ones(state.shape[:2], dtype=torch.int32, device=state.device)
        extended_mask = torch.cat([extended_mask, state_mask], dim=1)
        xs = torch.cat([x.reshape(B, T * mult, D), state], dim=1)
        extended_mask = self.get_extended_attention_mask(extended_mask, xs.shape)
        out = self.encoder.forward(xs, attention_mask=extended_mask)['last_hidden_state'][:, (mult-1):T*mult:mult].reshape(B, T, D)
        return self.head(out)
    

class SpliceLoss(nn.Module):

    def forward(self, logits: Tensor, labels: Tensor):
        B = logits.shape[0]
        pos_weight = torch.tensor([1.0, 100.0, 100.0])
        pos_weight_seq = pos_weight.repeat(B, 1).cuda()
        loss_fct = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_seq)
        loss = loss_fct(logits, labels)
        loss = loss.sum() / B if B != 0.0 else torch.tensor(0.0, device=logits.device)
        return loss
    

class SpliceLossFlat(nn.Module):

    def forward(self, logits: Tensor, labels: Tensor):
        B = logits.shape[0]
        pos_weight = torch.tensor([1.0, 100.0, 100.0], device=logits.device)
        pos_weight_seq = pos_weight.repeat(B, 1)
        loss_fct = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_seq)
        loss = loss_fct(logits, labels)
        return loss
