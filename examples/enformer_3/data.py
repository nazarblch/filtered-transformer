import random
import re

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from functools import reduce


class EnformerDataset(Dataset):
    """
    seq = left_context; BIN_SIZE x BINS_COUNT; right_context
    target: BINS_COUNT x TG_COUNT

    EnformerDataset splits seq and target on chunks with n_bins = `bins_per_sample`:
    ->
    chunk_seq := CLS bin_1 SEP bin_2 SEP bin_3 ... SEP
    target: bins_per_sample x TG_COUNT
    """
    TG_COUNT = 5313
    BINS_COUNT = 896
    BIN_SIZE = 128
    PAD = (196608 - BIN_SIZE * BINS_COUNT) // 2

    def __init__(self, tokenizer, path: str, max_seq_len: int = 512):
        self.h5_file = h5py.File(path, "r")
        self.h5_keys = np.asarray(list(self.h5_file.keys()))

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.n_records = len(self.h5_keys)
        self.n_chunks = 50

    def __len__(self):
        # each record in dataset is split on n_chunks, in total we have n_records x n_chunks samples in a dataset
        return self.n_records * self.n_chunks

    def __getitem__(self, idx):
        record_idx = idx // self.n_chunks
        k = self.h5_keys[record_idx]

        # it takes ~200ms to read single seq & target from h5file (in random order)
        seq = self.h5_file[k]['seq'][()].decode('utf-8')
        
        center = seq[EnformerDataset.PAD: -EnformerDataset.PAD]
        assert len(center) // self.BIN_SIZE == self.BINS_COUNT

        bins_seq = [center[
            i * self.BIN_SIZE: (i+1) * self.BIN_SIZE
        ] for i in range(self.BINS_COUNT)]

        encoded_bins = self.tokenizer.batch_encode_plus(bins_seq, add_special_tokens=False, return_attention_mask=False,
                                                        return_token_type_ids=False)['input_ids']

        right_pad = 0
        t = 0
        T = len(encoded_bins)
        while right_pad < 512:
            t += 1
            right_pad += len(encoded_bins[T - t])

        pos_1 = random.randint(0, T - t)
        pos_2 = pos_1 - 1
        right_pad = 0
        while right_pad < 512 and pos_2 < T:
            pos_2 += 1
            right_pad += len(encoded_bins[pos_2])

        active_bins = encoded_bins[pos_1:pos_2]
        
        sep = self.tokenizer.sep_token_id
        CLEN = 1022
        sample_token_ids = reduce(lambda b1, b2: b1 + b2, [b + [sep] for b in active_bins])
        left_context = reduce(lambda b1, b2: b1 + b2, [b for b in encoded_bins[:pos_1] + [[]]])[-CLEN:]
        right_context = reduce(lambda b1, b2: b1 + b2, [b for b in encoded_bins[pos_2:] + [[]]])[:CLEN]

        if len(left_context) < CLEN:
            left_seq = [seq[:EnformerDataset.PAD]]
            left_inputs = self.tokenizer.batch_encode_plus(left_seq, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'][0]
            left_context = (left_inputs + left_context)[-CLEN:]

        if len(right_context) < CLEN:
            right_seq = [seq[-EnformerDataset.PAD:]]
            rigth_inputs = self.tokenizer.batch_encode_plus(right_seq, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'][0]
            right_context = (right_context + rigth_inputs)[:CLEN]

        # print(len(left_context), len(sample_token_ids), len(right_context))

        sample_token_ids = np.array(left_context + sample_token_ids + right_context)

        token_type_ids = np.array([0] * len(sample_token_ids))
        attention_mask = np.array([1] * len(sample_token_ids))
        bins_mask = (sample_token_ids == self.tokenizer.sep_token_id).astype(bool)

        assert bins_mask.astype(np.int32).sum().item() == pos_2 - pos_1

        target = self.h5_file[k]['target'][pos_1:pos_2, :]


        return {'input_ids': sample_token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'bins_mask': bins_mask,
                'labels': target}

        
class TestEnformerDataset(Dataset):
    
    TG_COUNT = 5313
    BINS_COUNT = 896
    BIN_SIZE = 128
    PAD = (196608 - BIN_SIZE * BINS_COUNT) // 2

    def __init__(self, tokenizer, path: str, max_seq_len: int = 512):
        self.h5_file = h5py.File(path, "r")
        self.h5_keys = np.asarray(list(self.h5_file.keys()))

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.n_records = len(self.h5_keys)

    def __len__(self):
        # each record in dataset is split on n_chunks, in total we have n_records x n_chunks samples in a dataset
        return self.n_records

    def __getitem__(self, idx):
        record_idx = idx
        k = self.h5_keys[record_idx]

        # it takes ~200ms to read single seq & target from h5file (in random order)
        seq = self.h5_file[k]['seq'][()].decode('utf-8')
        
        center = seq[EnformerDataset.PAD: -EnformerDataset.PAD]
        assert len(center) // self.BIN_SIZE == self.BINS_COUNT

        bins_seq = [center[
            i * self.BIN_SIZE: (i+1) * self.BIN_SIZE
        ] for i in range(self.BINS_COUNT)]

        encoded_bins = self.tokenizer.batch_encode_plus(bins_seq, add_special_tokens=False, return_attention_mask=False,
                                                        return_token_type_ids=False)['input_ids']
        
        CLEN = 1022
        
        left_seq = [seq[:EnformerDataset.PAD]]
        left_inputs = self.tokenizer.batch_encode_plus(left_seq, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'][0]
              
        right_seq = [seq[-EnformerDataset.PAD:]]
        rigth_inputs = self.tokenizer.batch_encode_plus(right_seq, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'][0]

        target = self.h5_file[k]['target'][()] 
        
        T = len(encoded_bins)

        chunks = []
        pos_1 = 0
        pos_2 = -1

        while pos_2 < T:

            pos_2 = pos_1

            right_pad = 0
            while right_pad < 512 and pos_2 < T:
                right_pad += len(encoded_bins[pos_2])
                pos_2 += 1

            active_bins = encoded_bins[pos_1:pos_2]
            
            sep = self.tokenizer.sep_token_id
            sample_token_ids = reduce(lambda b1, b2: b1 + b2, [b + [sep] for b in active_bins])
            left_context = reduce(lambda b1, b2: b1 + b2, [b for b in encoded_bins[:pos_1] + [[]]])[-CLEN:]
            right_context = reduce(lambda b1, b2: b1 + b2, [b for b in encoded_bins[pos_2:] + [[]]])[:CLEN]

            if len(left_context) < CLEN:
                left_context = (left_inputs + left_context)[-CLEN:]

            if len(right_context) < CLEN:
                right_context = (right_context + rigth_inputs)[:CLEN]

            sample_token_ids = np.array(left_context + sample_token_ids + right_context)

            token_type_ids = np.array([0] * len(sample_token_ids))
            attention_mask = np.array([1] * len(sample_token_ids))
            bins_mask = (sample_token_ids == self.tokenizer.sep_token_id).astype(bool)

            assert bins_mask.astype(np.int32).sum().item() == pos_2 - pos_1

            pos_1 = pos_2

            chunks.append({'input_ids': sample_token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'bins_mask': bins_mask})
            
        return {"chunks": chunks, "labels": target}