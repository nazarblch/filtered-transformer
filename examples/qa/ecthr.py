from typing import Dict
from examples.qa.data import get_tokenized_dataset
from examples.qa.modules2 import DataFilter, MemUpMemoryImpl, Predictor, RobertaRT
from examples.qa.tasks import ECTHRBinaryClassificationTask
from memup.base import DataCollectorAppend, DataCollectorReplace, MemoryRollout, State
from metrics.accuracy import AccuracyMetric
import tasks
from transformers import RobertaTokenizer, RobertaModel
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter


task = ECTHRBinaryClassificationTask(("ecthr_binary", "bal", "facts", None, "violated_articles", 2),
                                                 base_path="/home/jovyan/nazar/ecthr_binary_bal")

dataset_dict = task.get_datasets()
tokenizer = RobertaTokenizer.from_pretrained(
    "roberta-base",
    cache_dir="/home/jovyan/cashe",
    use_fast=True,
    revision="main",
)

model = RobertaRT(RobertaModel.from_pretrained(
    'roberta-base',
    cache_dir="/home/jovyan/cashe",
    revision="main",
)).cuda()

predictor = Predictor(model.bert.config).cuda()

# weights = torch.load("/home/jovyan/models/etc_1_900.pt", map_location="cpu")
# model.load_state_dict(weights["mem"])
# predictor.load_state_dict(weights["pred"])

tokenized_dataset_dict = get_tokenized_dataset(
    task=task,
    dataset_dict=dataset_dict,
    tokenizer=tokenizer,
    max_seq_length=3300,
    padding_strategy=PaddingStrategy(PaddingStrategy.MAX_LENGTH),
    truncation_strategy=TruncationStrategy(TruncationStrategy.ONLY_FIRST),
    model_mode="cls"
)

train_data = tokenized_dataset_dict.get("train")
validation_data = tokenized_dataset_dict.get("validation")
test_data = tokenized_dataset_dict.get("test")

print(len(train_data), len(validation_data))
print(train_data[0].keys())
print(len(train_data[0]["input_ids"]))
print(len(train_data[1]["input_ids"]))
 
def collate_fn(batch):

    batch_pt = {}
        
    for k in ['input_ids', 'attention_mask', "label"]:
        batch_pt[k] = torch.stack(
            [torch.tensor(el[k]) for el in batch]
        )

    return batch_pt
        

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32, num_workers=8, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_data, shuffle=False, batch_size=128, num_workers=8, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=128, num_workers=8, collate_fn=collate_fn)

data_filter = DataFilter(tokenizer, 300)

memup_iter = MemoryRollout[Dict[str, torch.Tensor]](
    steps=2,
    memory=MemUpMemoryImpl(model),
    data_filter=data_filter,
    info_update=[]
)

opt = AdamW([
    {"params": model.bert.parameters(), "lr": 1e-6},
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": predictor.parameters(), "lr": 1e-5},
] , weight_decay=1e-5)


class DataCollectorTrain(DataCollectorAppend[Dict[str, torch.Tensor], Tensor]):
    def apply(self, data: Dict[str, Tensor], out: Tensor, state: State) -> Tensor:
        return state
    

class DataCollectorEval(DataCollectorReplace[Dict[str, Tensor], Tensor]):
    def apply(self, data: Dict[str, Tensor], out: Tensor, state: State) -> Tensor:
        return state


writer = SummaryWriter("/home/jovyan/pomoika/ect/1.5")
global_step = 0
batch_count = 0

for it in range(100):

    for batch in train_dataloader:
        batch_count += 1

        labels = batch["label"].cuda()

        state = torch.zeros(labels.shape[0], 30, 768, device=torch.device("cuda"))
        done = False
        info = {}

        model.train()
        predictor.train()

        print(it, batch_count, global_step)

        grad_acc_times = 5

        while not done:
            global_step += 1

            data_collector, state, info, done = memup_iter.forward(batch, state, info, DataCollectorTrain())
            states_seq = data_collector.result()
            # pred = predictor(torch.cat([states_seq[-1], last_state], 1))
            pred = predictor(states_seq[-1])
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

        if batch_count % 50 == 1:
            # torch.save({
            #     "mem": model.state_dict(),
            #     "pred": predictor.state_dict()
            # }, f"/home/jovyan/models/etc_1_{batch_count}.pt")

            print("EVAL length ", len(validation_data))

            with torch.no_grad():

                all_pred = []
                all_labels = []

                for batch in validation_dataloader:
                    labels = batch["label"].cuda()

                    state = torch.zeros(labels.shape[0], 30, 768, device=torch.device("cuda"))
                    done = False
                    info = {}

                    model.eval()
                    predictor.eval()

                    data_collector, last_state, _, _ = memup_iter.forward(batch, state, info, DataCollectorEval(), steps=1000)
                    pred = predictor(last_state)
                    loss = nn.CrossEntropyLoss()(pred, labels)
                    acc = AccuracyMetric()(pred, labels)

                    print(loss.item(), "acc=", acc)
                    writer.add_scalar("eval loss", loss.item(), global_step)

                    all_pred.append(pred.detach().cpu())
                    all_labels.append(labels.cpu())
                
                acc = AccuracyMetric()(torch.cat(all_pred), torch.cat(all_labels))
                print("eval acc", acc)
                writer.add_scalar("eval acc", acc, global_step)

            print("TEST length ", len(test_data))

            with torch.no_grad():

                all_pred = []
                all_labels = []

                for batch in test_dataloader:
                    labels = batch["label"].cuda()

                    state = torch.zeros(labels.shape[0], 30, 768, device=torch.device("cuda"))
                    done = False
                    info = {}

                    model.eval()
                    predictor.eval()

                    data_collector, last_state, _, _ = memup_iter.forward(batch, state, info, DataCollectorEval(), steps=1000)
                    pred = predictor(last_state)
                    loss = nn.CrossEntropyLoss()(pred, labels)
                    acc = AccuracyMetric()(pred, labels)

                    print(loss.item(), "acc=", acc)
                    writer.add_scalar("eval loss", loss.item(), global_step)

                    all_pred.append(pred.detach().cpu())
                    all_labels.append(labels.cpu())
                
                acc = AccuracyMetric()(torch.cat(all_pred), torch.cat(all_labels))
                print("test acc", acc)
                writer.add_scalar("test acc", acc, global_step)


