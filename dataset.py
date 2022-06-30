from typing import List, Dict

from torch.utils.data import Dataset
import torch

from utils import Vocab
import numpy as np


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
        ix_of_texts = self.vocab.encode_batch(tokenized_texts)
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

class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        tag_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.tag_mapping = tag_mapping
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.tag_mapping)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        # 1. Tokenize sentences into List<List<str>>
        tokenized_texts = [sample["tokens"]  for sample in samples]
        # 2. Convert sentences to ix and pad them to max_len
        ix_of_texts = self.vocab.encode_batch(tokenized_texts)
        ix_of_texts = [torch.LongTensor(ix_of_text) for ix_of_text in ix_of_texts]
        ix_of_texts = torch.stack(ix_of_texts,0)
        
        if "tags" in samples[0].keys():
            ## train and val
            # 1. convert tags into idx
            # taggeds: List<List<str>>
            taggeds = [sample["tags"] for sample in samples]
            taggeds = [ [self.tag2idx(tag) for tag in tagged] for tagged in taggeds]
            # 2. pad them to max_len
            # initialize tags 才能區分PAD tags
            to_len = max(len(tokenized_text) for tokenized_text in tokenized_texts)
            tags = (-1) * np.ones((len(taggeds), to_len))
            for i in range(len(taggeds)):
                tag_num = len(taggeds[i])
                tags[i][:tag_num] = taggeds[i]
            
            tags = torch.LongTensor(tags)
            return ix_of_texts, tags
        else:
            # test
            # ids: list of length batch_size
            ids = [sample["id"] for sample in samples]
            return ix_of_texts, ids

    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]