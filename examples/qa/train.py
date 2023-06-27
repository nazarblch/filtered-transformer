import random
import sys
import os
from typing import Dict

sys.path.append("/home/jovyan/filtered-transformer/")

from memup.base import DataCollectorAppend, DataCollectorReplace, MemoryRollout
from memup.loss import TS, LossModule, PredictorLossStateOnly
from metrics.accuracy import AccuracyMetric
from torch import Tensor, nn
from examples.qa.modules import DataFilter, MemUpMemoryImpl, Predictor, RobertaRT
from memup.base import DataCollectorAppend, MemoryRollout, State
from memup.loss import TS, LossModule, PredictorLossStateOnly
from metrics.accuracy import AccuracyMetric
from examples.qa.data import get_tokenized_dataset
from transformers import AutoConfig, AutoTokenizer
import transformers
import tasks
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def adjust_tokenizer(tokenizer):
    if isinstance(tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)) and \
            "gpt" in tokenizer.name_or_path:
        tokenizer.pad_token = tokenizer.eos_token


tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base",
    cache_dir="/home/jovyan/cashe",
    use_fast=True,
    revision="main",
)

adjust_tokenizer(tokenizer)


model = RobertaRT(RobertaModel.from_pretrained(
    'roberta-base',
    cache_dir="/home/jovyan/cashe",
    revision="main",
)).cuda()

predictor = Predictor(model.bert.config).cuda()

weights = torch.load("/home/jovyan/qa_1.180.pt", map_location="cpu")
model.load_state_dict(weights["mem"])
predictor.load_state_dict(weights["pred"])



task = tasks.get_task(task_args=tasks.TaskArguments(task_name="custom", task_base_path="/home/jovyan/quality_mc/"))
dataset_dict = task.get_datasets()

tokenized_dataset_dict = get_tokenized_dataset(
    task=task,
    dataset_dict=dataset_dict,
    tokenizer=tokenizer,
    max_seq_length=8000,
    padding_strategy=PaddingStrategy(PaddingStrategy.MAX_LENGTH),
    truncation_strategy=TruncationStrategy(TruncationStrategy.ONLY_FIRST),
    model_mode="mc",
)

train_data = tokenized_dataset_dict.get("train")
test_data = tokenized_dataset_dict.get("validation")

def collate_fn(batch):

    batch_pt = {}
        
    for k in ['input_ids', 'attention_mask', "label", 'input_part_token_start_idx']:
        batch_pt[k] = torch.stack(
            [torch.tensor(el[k]) for el in batch]
        )

    return batch_pt


train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32, num_workers=8, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=128, num_workers=8, collate_fn=collate_fn)

data_filter = DataFilter(tokenizer, 200)

memup_iter = MemoryRollout[Dict[str, Tensor]](
    steps=2,
    memory=MemUpMemoryImpl(model),
    data_filter=data_filter,
    info_update=[]
)

opt = AdamW([
    {"params": model.bert.parameters(), "lr": 1e-6},
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": predictor.parameters(), "lr": 1e-5},
] , weight_decay=1e-4)


class DataCollectorTrain(DataCollectorAppend[Dict[str, Tensor], Tensor]):
    def apply(self, data: Dict[str, Tensor], out: Tensor, state: State) -> Tensor:
        return state
    

class DataCollectorEval(DataCollectorReplace[Dict[str, Tensor], Tensor]):
    def apply(self, data: Dict[str, Tensor], out: Tensor, state: State) -> Tensor:
        return state


writer = SummaryWriter("/home/jovyan/pomoika/qa/1.17")
global_step = 0
batch_count = 0

for it in range(100):

    for batch in train_dataloader:
        batch_count += 1

        labels = batch["label"].cuda()

        state = torch.zeros(labels.shape[0] * 4, 30, 768, device=torch.device("cuda"))
        done = False
        info = {}

        model.train()
        predictor.train()

        with torch.no_grad():
            _, last_state, _, _ = memup_iter.forward(batch, state, {}, DataCollectorEval(), 100)

        print(it, batch_count, global_step)

        grad_acc_times = 5

        while not done:
            global_step += 1

            data_collector, state, info, done = memup_iter.forward(batch, state, info, DataCollectorTrain())
            states_seq = data_collector.result()
            pred = predictor(torch.cat([states_seq[-1], last_state], 1))
            # pred = predictor(torch.cat(states_seq))
            loss = nn.CrossEntropyLoss()(pred, labels)
            acc = AccuracyMetric()(pred, labels)

            print(pred.argmax(-1).reshape(-1).cpu().numpy())
            
            print(loss.item(), "acc=", acc)
            writer.add_scalar("loss", loss.item(), global_step)
            writer.add_scalar("acc", acc, global_step)

            (loss / grad_acc_times).backward()

            if global_step % grad_acc_times == 0:
                opt.step()
                opt.zero_grad()

        if batch_count % 30 == 0:
            torch.save({
                "mem": model.state_dict(),
                "pred": predictor.state_dict()
            }, f"/home/jovyan/qa_4.{batch_count}.pt")

            print("EVAL length ", len(test_data))

            with torch.no_grad():

                all_pred = []
                all_labels = []

                for batch in test_dataloader:
                    labels = batch["label"].cuda()

                    state = torch.zeros(labels.shape[0] * 4, 30, 768, device=torch.device("cuda"))
                    done = False
                    info = {}

                    model.eval()
                    predictor.eval()

                    data_collector, _, _, _ = memup_iter.forward(batch, state, info, DataCollectorEval(), steps=1000)
                    states_seq = data_collector.result()
                    pred = predictor(torch.cat([states_seq[-1], states_seq[-1]], 1))
                    loss = nn.CrossEntropyLoss()(pred, labels)
                    acc = AccuracyMetric()(pred, labels)

                    print(loss.item(), "acc=", acc)
                    writer.add_scalar("eval loss", loss.item(), global_step)

                    all_pred.append(pred.detach().cpu())
                    all_labels.append(labels.cpu())
                
                acc = AccuracyMetric()(torch.cat(all_pred), torch.cat(all_labels))
                print("final acc", acc)
                writer.add_scalar("eval acc", acc, global_step)






        
