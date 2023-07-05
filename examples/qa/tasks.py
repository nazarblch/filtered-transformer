import os
from abc import abstractmethod
from dataclasses import dataclass, field
import numpy as np
import datasets
import transformers
from io_utils import read_json, read_jsonl
import datasets
import transformers
import zipfile
import wget


class Task:

    @property
    @abstractmethod
    def num_choices(self) -> int:
        raise NotImplementedError()

    @property
    def drop_columns(self) -> list:
        """Returns list of columns to drop when tokenizing
        (Not really necessary, just reduces clutter in the batch objects)

        Don't include any of:
            label
            context
            query
            option_*

        :return: list columns to drop
        """
        return []

    @abstractmethod
    def standardize_examples(self, examples) -> dict:
        """Called by (batched) dataset method to convert data to standard format
        Output is a dict of lists, with the following types
            - context: str
            - query: str
            - label: int
            - option_[0..NUM_CHOICES]: str

        Ultimately, examples will be formatted as:
            context + query + option
        or
            context + [sep] + query + option

        with NO SPACES, so adjust accordingly (e.g. prepending space to query/options)

        :return: dict of lists
        """
        raise NotImplementedError()

    @abstractmethod
    def get_datasets(self) -> dict:
        """Returns dict (or dict-like) of datasets, with keys:
            train
            validation
            test

        :return: dict[str, Dataset]
        """
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def compute_metrics(self, p: transformers.EvalPrediction):
        if isinstance(p, dict):
            return p['eval_accuracy']
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)
        
        if preds.ndim < 3:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        else:
            label_ids = p.label_ids
            total = 0
            num_correct = 0
            for idx, ex_labels in enumerate(label_ids):
                ex_labels[ex_labels == -100] = 1
                total += 1
                if (ex_labels == preds[idx]).all():
                    num_correct += 1
            return {'accuracy': num_correct / total}


class CosmosQATask(Task):
    @property
    def num_choices(self) -> int:
        return 4

    @property
    def drop_columns(self) -> list:
        return ["question", "answer0", "answer1", "answer2", "answer3"]

    @classmethod
    def standardize_examples(cls, examples):
        result = {
            "context": examples["context"],
            "query": prepend_space(examples["question"]),
        }
        for i in range(4):
            result[f"option_{i}"] = prepend_space(examples[f"answer{i}"])
        return result

    def get_datasets(self) -> dict:
        return datasets.load_dataset("cosmos_qa")
    
    
class RaceTask(Task):
    def get_datasets(self) -> dict:
        return datasets.load_dataset("race", "all")
    
    @classmethod
    def standardize_examples(cls, examples):
        result = {
            "context": examples["article"],
            "query": prepend_space(examples["question"]),
        }
        for i in range(4):
            result[f"option_{i}"] = prepend_space([ex_options[i] for ex_options in examples["options"]])
        label_mappings = {"A": 0, "B": 1, "C": 2, "D": 3}
        result["label"] = [label_mappings[ex_answer] for ex_answer in examples["answer"]]
        return result
    
    @property
    def drop_columns(self) -> list:
        return ["question", "article", "options", "answer"]
    
    @property
    def num_choices(self) -> int:
        return 4

class HyperpartisanTask(Task):
    def __init__(self, base_path,):
        self.base_path = base_path

    @classmethod
    def standardize_examples(cls, examples):
        # jsonl data should already be preformatted to have keys
        #    context
        #    query
        #    label
        #    option_*
        return examples

    def get_datasets(self) -> dict:
        phases = ["train", "dev", "test"]

        dataset_dict = {}
        for phase in phases:
            phase_path = os.path.join(self.base_path, f"{phase}.jsonl")
            # phase_path = os.path.join(self.base_path, f"quality_example.jsonl")
            if not os.path.exists(phase_path):
                continue
            dataset_dict[phase] = datasets.load_dataset(
                "json",
                data_files=phase_path,
            )["train"]  # <- yes this is weird

        return dataset_dict

    @classmethod
    def create_from_path(cls, base_path):
        return cls(
            base_path=base_path,
        )


class CustomJSONLTask(Task):
    def __init__(self, base_path, num_choices, drop_columns=None):
        self.base_path = base_path
        self._drop_columns = drop_columns if drop_columns else []
        self._num_choices = num_choices

    @property
    def drop_columns(self) -> list:
        return self._drop_columns

    @property
    def num_choices(self) -> int:
        return self._num_choices

    @classmethod
    def standardize_examples(cls, examples):
        # jsonl data should already be preformatted to have keys
        #    context
        #    query
        #    label
        #    option_*
        return examples

    def get_datasets(self) -> dict:
        phases = ["train", "validation", "test"]
        dataset_dict = {}
        for phase in phases:
            phase_path = os.path.join(self.base_path, f"{phase}.jsonl")
            #phase_path = os.path.join(self.base_path, f"quality_example.jsonl")
            if not os.path.exists(phase_path):
                continue
            dataset_dict[phase] = datasets.load_dataset(
                "json",
                data_files=phase_path,
            )["train"]  # <- yes this is weird
        return dataset_dict

    @classmethod
    def create_from_path(cls, base_path):
        config = read_json(os.path.join(base_path, "config.json"))
        return cls(
            base_path=base_path,
            num_choices=config["num_choices"],
            drop_columns=config.get("drop_columns", []),
        )

def prepend_space(list_of_strings: list) -> list:
    return [" " + x for x in list_of_strings]


@dataclass
class TaskArguments:
    task_name: str = field(
        metadata={"help": "Task name (e.g. CosmosQA, CustomJSONLTask)"}
    )
    task_base_path: str = field(
        metadata={"help": "Path to data from CustomJSONLTask"},
        default=None,
    )


def get_task(task_args: TaskArguments):
    if task_args.task_name == "custom":
        return CustomJSONLTask.create_from_path(base_path=task_args.task_base_path)
    elif task_args.task_name == "hp_seq_cls":
        return HyperpartisanTask.create_from_path(base_path=task_args.task_base_path)

    task_dict = {
        "cosmosqa": CosmosQATask,
        "race": RaceTask,
    }
    return task_dict[task_args.task_name]()


class HFClassificationTask(Task):
    def __init__(self, task_args):
        """Only for tasks with two inputs!
        (and mostly for debug purposes)
        """
        self.task_name, self.subset, self.key1,\
        self.key2, self.label, self._num_labels = task_args
        self._num_choices = self._num_labels
    
    def get_datasets(self) -> dict:
        if self.subset is not None:
            return datasets.load_dataset(self.task_name, self.subset)
        return datasets.load_dataset(self.task_name)
    
    def standardize_examples(self, examples):
        result = {
            "context": examples[self.key1],
            "label": list(map(int, examples[self.label])),
        }
        if self.key2 is not None:
            result[self.key2] = examples[self.key2]
        return result
    
    @property
    def drop_columns(self) -> list:
        return []
    
    @property
    def num_choices(self) -> int:
        return self._num_choices

    @property
    def num_labels(self) -> int:
        return self._num_labels


class ECTHRBinaryClassificationTask(HFClassificationTask):
    def __init__(self, task_args, base_path):
        super().__init__(task_args)
        self.base_path = base_path

    def get_datasets(self) -> dict:
        # load data if needed
        data_link = "https://archive.org/download/ECtHR-NAACL2021/dataset.zip"
        files = ["train.jsonl", "dev.jsonl", "test.jsonl"]
        os.makedirs(self.base_path, exist_ok=True)
        file_pathes = [os.path.join(self.base_path, fname) for fname in files]
        if not(all([os.path.isfile(file) for file in file_pathes])):
            fname = wget.download(data_link, out=self.base_path)
            with zipfile.ZipFile(fname, "r") as zip_ref:
                zip_ref.extractall(self.base_path)
        phases = ["train", "validation", "test"]
        dataset_dict = {}
        #from transformers import AutoTokenizer
        #tokenizer = AutoTokenizer.from_pretrained(
        #    "roberta-base",
        #)
        for phase, file in zip(phases, files):
            phase_path = os.path.join(self.base_path, file)
            data = read_jsonl(phase_path)
            data_dict = {"context": list(map(" ".join, [el[self.key1] for el in data])),
                         "label": list(map(int, [len(el[self.label]) > 0 for el in data]))}
            dataset_dict[phase] = datasets.Dataset.from_dict(data_dict)
            if self.subset == "bal":
                # undersample all
                dataset_dict[phase] = self.resample_dataset(dataset_dict, phase)
            elif self.subset == "bal_train":
                # undersample only train
                if phase == "train":
                    dataset_dict[phase] = self.resample_dataset(dataset_dict, phase)
            print(f"Stats for {phase}: {np.unique(dataset_dict[phase]['label'], return_counts=True)}")
            #lens = [len(tokenizer(el)["input_ids"]) for el in dataset_dict[phase]['context']]
            #print(f"Samples stats for {phase}: min words: {np.min(lens)}, max words {np.max(lens)}, mean words {np.mean(lens)}, 95 percentile: {np.percentile(lens, 95)} ")
        return dataset_dict

    def resample_dataset(self, dataset, split="train"):
        # balance classes
        # fix seeds for reproducibility
        np.random.seed(42)
        negatives = dataset[split].filter(lambda el: el["label"] == 0)
        positives = dataset[split].filter(lambda el: el["label"] == 1)
        pos_indices = np.arange(len(positives))
        indices = np.random.choice(pos_indices,
                                   len(negatives),
                                   replace=False)
        return datasets.concatenate_datasets([negatives, positives.select(indices)])
        

    def standardize_examples(self, examples):
        return examples


