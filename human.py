import numpy as np
import torch
from gena_lm.modeling_bert import BertForSequenceClassification, BertModel
from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
model: BertModel = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base').bert.cuda()

data = pd.read_csv("/home/nazar/PycharmProjects/GENA_LM/downstream_tasks/promoter_prediction/hg38_len_300_promoters_dataset.csv")

X = np.asarray(data["sequence"].tolist())
Y = np.asarray(data["promoter_presence"].tolist())

t = tokenizer(X[0])

output = model.forward(input_ids=torch.tensor(t['input_ids'])[None,].cuda(), attention_mask=torch.tensor(t['attention_mask'])[None,].cuda(), output_hidden_states=True)
print(output["last_hidden_state"].shape)