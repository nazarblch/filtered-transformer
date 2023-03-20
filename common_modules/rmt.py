import math
from typing import Dict, Tuple
from tokenizers import Tokenizer
from torch import nn
from torch import Tensor
from common_modules.transformers import RecurrentOutput, RecurrentTransformer
import torch
import torch.nn.functional as F


class RecurrentTransformerWithStateEmbedding(RecurrentTransformer):
    def __init__(self, base_model: nn.Module, num_mem_tokens: int, tokenizer: Tokenizer):
        super().__init__()
        self.model = base_model
        self._extract_special_tokens(tokenizer)
        self._extend_word_embeddings(num_mem_tokens)
        self.memory_position = range(1, 1 + num_mem_tokens)

    def _init_memory(self) -> Tensor:
        mem_token_ids = self.mem_token_ids
        memory = self.model.base_model.embeddings.word_embeddings(mem_token_ids)
        return memory
    
    def init_state(self, batch_size) -> Tensor:
        memory = self._init_memory()
        return memory.repeat(batch_size, 1, 1)

    def _extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.register_buffer('cls_token', torch.tensor([tokenizer.cls_token_id]))
        self.register_buffer('sep_token', torch.tensor([tokenizer.sep_token_id]))

    def _extend_word_embeddings(self, num_mem_tokens):
        vocab_size: int = self.model.base_model.embeddings.word_embeddings.weight.shape[0]
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)
        # self.model.embeddings = self.model.base_model.embeddings.word_embeddings

    def _extend_input(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if input_ids[0][0].item() == self.cls_token.item():
           input_ids = input_ids[:, 1:] 
           attention_mask = attention_mask[:, 1:]
           token_type_ids = token_type_ids[:, 1:]
        
        prefix = torch.cat([self.cls_token, self.mem_token_ids, self.sep_token])[None, :].repeat(input_ids.shape[0], 1)
        input_ids = torch.cat([prefix, input_ids], dim=1)
        attention_mask = torch.cat([torch.ones_like(prefix), attention_mask], dim=1)
        token_type_ids = torch.cat([torch.zeros_like(prefix), token_type_ids], dim=1)

        return input_ids, attention_mask, token_type_ids

    def forward(self, x: Dict[str, Tensor], state: Tensor) -> RecurrentOutput:

        assert self.num_mem_tokens == state.shape[1]

        input_ids, attention_mask, token_type_ids = self._extend_input(x["input_ids"], x["attention_mask"], x["token_type_ids"])
        empty_mask = x["attention_mask"].type(torch.int32).sum(1) == 0
        
        inputs_embeds = self.model.base_model.embeddings.word_embeddings(input_ids)
        inputs_embeds[:, self.memory_position] = state

        seg_kwargs = {}
        seg_kwargs['output_hidden_states'] = True
        seg_kwargs['inputs_embeds'] = inputs_embeds
        seg_kwargs['attention_mask'] = attention_mask
        seg_kwargs['token_type_ids'] = token_type_ids

        so = self.model(**seg_kwargs)
        new_state = so.hidden_states[-1][:, self.memory_position]
        new_state[empty_mask] = state[empty_mask]
        out = so.hidden_states[-1][:, self.num_mem_tokens + 2:]

        assert out.shape[1] == x["input_ids"].shape[1]

        return RecurrentOutput(out, new_state)
