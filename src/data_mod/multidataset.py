import os
import transformers
import yaml
import torch
from torch.utils.data import Sampler, Dataset
import time
import random
import math  # Import the math module
from .multimode_dataset import MultiModeLazyDataset

"""
Configuration YAML example:

laion400m:
    data_path: path/to/my/data
    max_samples: 10000
    simple: False
    supported_tasks: ['captioning']
pixmo:
    data_path: path/to/my/data
    max_samples: 10000
    simple: False
    supported_tasks: ['captioning']
mammoth:
    data_path: path/to/my/data
    max_samples: 10000
    simple: False
    supported_tasks: ['captioning']
recipeqa:
    data_path: path/to/my/data
    max_samples: 10000
    simple: False
    wrap_img_tokens: True
    supported_tasks: ['retrieval']

"""
class MultiDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):
        self.tokenizer = tokenizer
        self.debug = debug
        self.datasets = {'retrieval': [], 'captioning': []}
        self.load_datasets(data_path)
        assert len(self.datasets['retrieval']) == 0 or len(self.datasets['captioning']) == 0, f"For now, only 1 task type is supported :("
        self.used_task = 'retrieval' if len(self.datasets['retrieval']) > 0 else 'captioning'
        self.dataset_order = []
        self.index_order = []
        for i, dataset in enumerate(self.datasets[self.used_task]):
            self.dataset_order.extend([i] * len(dataset))
            self.index_order.extend(list(range(len(dataset))))
        
        combined = list(zip(self.dataset_order, self.index_order))
        random.shuffle(combined)
        self.dataset_order, self.index_order = zip(*combined)
        self.dataset_order = list(self.dataset_order)
        self.index_order = list(self.index_order)

    def load_datasets(self, data_path):
        with open(data_path, 'r') as file:
            config = yaml.safe_load(file)

        for dataset_name, dataset_config in config.items():
            print(f"Loading {dataset_name} ....")
            start_time = time.time()
            assert len(dataset_config.get('supported_tasks', [])) == 1, f"For now only 1 task is supported for each dataset, got {dataset_config.get('supported_tasks', [])}"
            task_type = dataset_config.get('supported_tasks', [])[0]

            dataset = MultiModeLazyDataset(
                tokenizer=self.tokenizer,
                **dataset_config
            )
            self.datasets[task_type].append(dataset)
            print(f"Done loading {dataset_name}, took {time.time() - start_time} seconds. It has {len(dataset)} samples")
        print(f"Multi dataset loaded, total samples: {self.__len__()}")


    def __len__(self):
        return sum(len(ds) for task_datasets in self.datasets.values() for ds in task_datasets)
    
    def __getitem__(self, idx):
        """Iterate every dataset to find the one at index idx"""
        #   for task_type in ['retrieval', 'captioning']:
        #       for dataset_idx, dataset in enumerate(self.datasets[task_type]):
        #           if idx < len(dataset):
        #               return dataset[idx]
        #           idx -= len(dataset)
        #   raise IndexError("Index out of range")
        return self.datasets[self.used_task][self.dataset_order[idx]][self.index_order[idx]]


# ========== CUSTOM SAMPLER ============


class MultiTaskSampler(Sampler):
    def __init__(self, dataset: MultiDataset, num_replicas=None, rank=None, shuffle=True, seed=0, batch_size = 1):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        # Calculate num_samples per replica, accounting for potential rounding
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.batch_size = batch_size  # Add batch_size
        # self._create_batches()

    def _create_batches(self):
        """Creates batches of indices, grouped by task."""
        all_batches = []

        for task in ['captioning', 'retrieval']:
            task_indices = []
            # Get indices for the current task
            offset = 0
            for i in range(len(self.dataset.datasets[task])):
                for j in range(len(self.dataset.datasets[task][i])):
                  task_indices.append((task,i,offset+j))
                offset += len(self.dataset.datasets[task][i])

            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                random.Random(g.seed()).shuffle(task_indices)

            # Create task-specific batches
            task_batches = [task_indices[i:i + self.batch_size] for i in range(0, len(task_indices), self.batch_size)]
            all_batches.extend(task_batches)  # Add to the overall list

        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(all_batches)
        self.batches = all_batches

    def __iter__(self):
      self.epoch = 0

      while True: # Make the sampler infinite
        self._create_batches()  # Re-create batches each epoch
        # Get batches for this rank
        batches_for_rank = self.batches[self.rank:self.total_size:self.num_replicas]


        for batch in batches_for_rank:
          yield [idx for task_type, dataset_idx, idx in batch]


        self.epoch += 1

    def __len__(self):
        return self.num_samples  # Number of samples for *this* replica


    def set_epoch(self, epoch: int) -> None:
      self.epoch = epoch