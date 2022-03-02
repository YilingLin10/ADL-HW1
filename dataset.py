from typing import List, Dict

from torch.utils.data import Dataset
import torch

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        # 1. Tokenize sentences 
        tokenized_texts = [sample["text"].split(" ")  for sample in samples]
        # 2. Convert sentences to ix and pad them to max_len
        ix_of_texts = self.vocab.encode_batch(tokenized_texts, to_len=self.max_len)
        ix_of_texts = [torch.LongTensor(ix_of_text) for ix_of_text in ix_of_texts]
        ix_of_texts = torch.stack(ix_of_texts,0)
        
        if "intent" in samples[0].keys():
            ## train and val
            # labels: [batch_size]
            labels = torch.LongTensor([self.label2idx(sample["intent"]) for sample in samples])
            return ix_of_texts, labels
        else:
            # test
            # ids: list of length batch_size
            ids = [sample["id"] for sample in samples]
            return ix_of_texts, ids

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
