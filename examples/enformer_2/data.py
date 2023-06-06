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
    # BLOCK_SIZE = 510

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

        sep = self.tokenizer.sep_token_id
        sample_token_ids = reduce(lambda b1, b2: b1 + b2, [b + [sep] for b in encoded_bins])
        sample_token_ids = np.array(sample_token_ids)

        token_type_ids = np.array([0] * len(sample_token_ids))
        attention_mask = np.array([1] * len(sample_token_ids))
        bins_mask = (sample_token_ids == self.tokenizer.sep_token_id).astype(bool)

        center_dict = {'input_ids': sample_token_ids,
                       'token_type_ids': token_type_ids,
                       'attention_mask': attention_mask,
                       'bins_mask': bins_mask,
                       'labels': target}
        
        BIN_SIZE = self.BIN_SIZE * 4
        left_seq = [left[i * BIN_SIZE: (i+1) * BIN_SIZE] for i in range(len(left) // BIN_SIZE)]
        right_seq = [right[i * BIN_SIZE: (i+1) * BIN_SIZE] for i in range(len(right) // BIN_SIZE)]

        left_inputs = self.tokenizer.batch_encode_plus(left_seq, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids']
        right_inputs = self.tokenizer.batch_encode_plus(right_seq, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids']
        # BS = EnformerDataset.BLOCK_SIZE
        left_inputs = np.asarray(reduce(lambda b1, b2: b1 + b2, [li + [sep] for li in left_inputs]))
        right_inputs = np.asarray(reduce(lambda b1, b2: b1 + b2, [ri + [sep] for ri in right_inputs]))
        
        left_bins_mask = (left_inputs == self.tokenizer.sep_token_id).astype(bool)
        right_bins_mask = (right_inputs == self.tokenizer.sep_token_id).astype(bool)

        left_dict = {'input_ids': left_inputs,
                     'token_type_ids': np.array([0] * len(left_inputs)),
                     'attention_mask': np.array([1] * len(left_inputs)),
                     'bins_mask': left_bins_mask}
        
        right_dict = {'input_ids': right_inputs,
                     'token_type_ids': np.array([0] * len(right_inputs)),
                     'attention_mask': np.array([1] * len(right_inputs)),
                     'bins_mask': right_bins_mask}
        
        return {
            "left": left_dict,
            "center": center_dict,
            "right": right_dict
        }
