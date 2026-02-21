from enum import Enum
import os

from .mmplanllm_dataset import MMPlanLLMDataset
from .multidataset import MultiDataset
from .multimode_dataset import LazySupervisedDataset, MultiModeLazyDataset


class DatasetType(str, Enum):
    MMPLANLLM_DATASET = "mmplanllm_dataset"
    MULTIMODE_DATASET = "multimode_dataset"
    MULTI_DATASET = "multi_dataset"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, DatasetType):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False


TYPE_TO_DATASET_CLASS = {
    DatasetType.MMPLANLLM_DATASET.value: MMPlanLLMDataset,
    DatasetType.MULTIMODE_DATASET.value: MultiModeLazyDataset,
    DatasetType.MULTI_DATASET.value: MultiDataset
}
