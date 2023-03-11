from functools import reduce
from itertools import chain
import h5py
import numpy as np
from torch.utils.data import Dataset


class EnformerDataset(Dataset):
    
    TG_COUNT = 5313
    BINS_COUNT = 896
    BIN_SIZE = 128
    PAD = (196608 - BIN_SIZE * BINS_COUNT) // 2

    def __init__(self, tokenizer, path: str, remove_context=True):
        self.h5_file = h5py.File(path, "r")
        self.h5_keys = np.asarray(list(self.h5_file.keys()))

        self.tokenizer = tokenizer
        self.remove_context = remove_context
        self.n_records = len(self.h5_keys)

    def __len__(self):
        return self.n_records

    def __getitem__(self, idx):
        k = self.h5_keys[idx]

        seq = self.h5_file[k]['seq'][()].decode('utf-8')
        target = self.h5_file[k]['target'][()]

        center = seq[EnformerDataset.PAD: -EnformerDataset.PAD]
        left, right = seq[:EnformerDataset.PAD], seq[-EnformerDataset.PAD:]
        
        assert len(center) // self.BIN_SIZE == self.BINS_COUNT

        # collect bins
        bins_seq = [center[
            i * self.BIN_SIZE: (i+1) * self.BIN_SIZE
        ] for i in range(self.BINS_COUNT)]

        encoded_bins = self.tokenizer.batch_encode_plus(bins_seq, add_special_tokens=False, return_attention_mask=False,
                                                        return_token_type_ids=False)['input_ids']

        # CLS left SEP bin_1 SEP bin_2 SEP bin_3 SEP ... bin_n SEP right SEP
        sep = self.tokenizer.sep_token_id
        sample_token_ids = [self.tokenizer.cls_token_id] + reduce(lambda b1, b2: b1 + b2, [b + [sep]*2 for b in encoded_bins])
        sample_token_ids = np.array(sample_token_ids)

        token_type_ids = np.array([0] * len(sample_token_ids))
        attention_mask = np.array([1] * len(sample_token_ids))
        bins_mask = (sample_token_ids == self.tokenizer.sep_token_id).astype(bool)

        center_dict = {'input_ids': sample_token_ids,
                       'token_type_ids': token_type_ids,
                       'attention_mask': attention_mask,
                       'bins_mask': bins_mask,
                       'labels': target}
        
        left_inputs = self.tokenizer.batch_encode_plus([left], add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'][0]
        right_inputs = self.tokenizer.batch_encode_plus([right], add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids'][0]

        left_dict = {'input_ids': np.array(left_inputs),
                     'token_type_ids': np.array([0] * len(left_inputs)),
                     'attention_mask': np.array([1] * len(left_inputs))}
        
        right_dict = {'input_ids': np.array(right_inputs),
                     'token_type_ids': np.array([0] * len(right_inputs)),
                     'attention_mask': np.array([1] * len(right_inputs))}
        
        return {
            "left": left_dict,
            "center": center_dict,
            "right": right_dict
        }
