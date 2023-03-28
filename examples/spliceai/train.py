import torch
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
import numpy as np
import sys
sys.path.append("/home/jovyan/filtered-transformer/")
from examples.spliceai.data import SpliceAIDataset
from gena_lm.modeling_bert import BertForSequenceClassification, BertForTokenClassification
from typing import Dict
from torch import Tensor
from data_filters.sliding_window import SlidingWindowFilterDict
from memup.base import MemoryRollout
from examples.spliceai.modules import ContextCollector, MemUpMemoryImpl, BertForSpliceAI, DataCollectorTrain, Predictor
from memup.preproc import IncrementStep
from common_modules.pos_encoding import PositionalEncoding2
from data_filters.top_errors import InputTarget, InputTargetMask, TopErrorsFilter, TopErrorsFilterWithMask
from memup.accumulator import Accumulator
from torch import nn
import os
from memup.loss import LossModule, PredictorLossWithContext
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
os.environ["TOKENIZERS_PARALLELISM"] = "true"


tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/filtered-transformer/data/tokenizers/t2t_1000h_multi_32k/")
data_path = "/home/jovyan/splice/train.csv.gz"
train_dataset = SpliceAIDataset(data_path, tokenizer, max_seq_len=4096, targets_offset=5000, targets_len=5000)

data_filter = SlidingWindowFilterDict(512, pad_fields={}, padding=0, skip_fields={"length"})

model_cfg = AutoConfig.from_pretrained('/home/jovyan/filtered-transformer/data/configs/L12-H768-A12-V32k-preln.json')
model = BertForSpliceAI(config=model_cfg)

bert = BertForTokenClassification(config=model_cfg)
ckpt_path = '/home/jovyan/splice/model_1900000.pth'
checkpoint = torch.load(ckpt_path, map_location='cpu')
bert.load_state_dict(checkpoint["model_state_dict"], strict=False)

model.bert = bert.bert
model = model.cuda()
model.train()

predictor = Predictor(model_cfg).cuda()
predictor.train()

optimizer = AdamW([
    {"params": model.bert.parameters(), "lr": 5e-5},
    {"params": model.encoder.parameters(), "lr": 5e-5},
    {"params": predictor.parameters(), "lr": 5e-5},
] , weight_decay=1e-5)


memup_iter = MemoryRollout[Dict[str, torch.Tensor]](
    steps=2,
    memory=MemUpMemoryImpl(model),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

mem_acc = Accumulator(model, decay=0.9)
pred_acc = Accumulator(predictor, decay=0.9)

errors_filter = TopErrorsFilterWithMask(303, (200, 300), 
                                        pred_acc, 
                                        nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([1.0, 100.0, 100.0])), 
                                        is_random=True)

memup_iter_acc = MemoryRollout[Dict[str, torch.Tensor]](
    steps=1000,
    memory=MemUpMemoryImpl(mem_acc.get_module()),
    data_filter=data_filter,
    info_update=[IncrementStep()]
)

predictor_loss = PredictorLossWithContext(predictor, [
        LossModule(nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 100.0, 100.0]).cuda()), "BCE", 1.0),
], cur_step_loss_coef=1)

pos_encoder = PositionalEncoding2(768, 0, 4096)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=24, num_workers=4)

writer = SummaryWriter("/home/jovyan/pomoika/splice1.4")
global_step = 0

for it, batch in enumerate(train_dataloader):

        model.train()
        predictor.train()
        
        info = {}
        done = False
        print()
        B, T = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
        state = torch.zeros(B, 100, model_cfg.hidden_size, device=torch.device("cuda:0"))

        batch["length"] = T
        batch["labels_mask"] = batch["labels_mask"].type(torch.bool)
        batch["positions"] = pos_encoder.forward(torch.zeros(1, T, 768)).expand(B, T, 768)
        
        with torch.no_grad():
            context_collector, last_state, _, _ = memup_iter_acc.forward(batch, state, {}, ContextCollector())
            context, c_mask, c_labels = context_collector.result()
            print("context", context.shape)
            last_state = last_state.cuda()
            assert context.shape[1] == c_mask.shape[1]

            # for j in range(0, context.shape[1], 81):
            #     mask = c_mask[:, j:j+81].cuda()
            #     pred_j = pred_acc(context[:, j:j+81].cuda(), last_state, mask).cpu()
            #     tg_j = c_labels[:, j:j+81]
                
            selected_data, _ = errors_filter.forward(InputTargetMask(context, c_labels, c_mask, context.shape[1]), last_state, {})
            
        while not done:
            global_step += 1

            data_collector, state, info, done = memup_iter.forward(batch, state, info, DataCollectorTrain())
            
            optimizer.zero_grad()
            loss = predictor_loss.forward(data_collector, info, selected_data, last_state)
            
            print(it, loss.item())
            print(info["losses"])
            loss.backward()
            optimizer.step()

            for name, val in info["losses"].items():
                writer.add_scalar(f"train/{name}", val, global_step)

            if global_step % 1000 == 0:
                    torch.save({
                        "mem": model.state_dict(),
                        "pred": predictor.state_dict(),
                        "mem_acc": mem_acc.get_module().state_dict(),
                        "pred_acc": pred_acc.get_module().state_dict()
                    }, "/home/jovyan/splice_1.pt")

        mem_acc.accumulate()
        pred_acc.accumulate()
        

               




