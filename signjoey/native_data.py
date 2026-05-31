# coding: utf-8
"""
Native (torchtext-free) data loading and batching.

This exists to make the project runnable on modern PyTorch builds (e.g. macOS MPS),
where torchtext's legacy `Field`/`BucketIterator` APIs are no longer available.
"""

from __future__ import annotations

import gzip
import os
import pickle
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from signjoey.vocabulary import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    Vocabulary,
    build_vocab,
)


def _load_dataset_file(filename: str):
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def _merge_samples(annotation_files: Sequence[str]) -> List[Dict]:
    samples: Dict[str, Dict] = {}
    for annotation_file in annotation_files:
        for sample in _load_dataset_file(annotation_file):
            seq_id = sample["name"]
            if seq_id in samples:
                prev = samples[seq_id]
                assert prev["name"] == sample["name"]
                assert prev["signer"] == sample["signer"]
                assert prev["gloss"] == sample["gloss"]
                assert prev["text"] == sample["text"]
                prev["sign"] = torch.cat([prev["sign"], sample["sign"]], axis=1)
            else:
                samples[seq_id] = {
                    "name": sample["name"],
                    "signer": sample["signer"],
                    "gloss": sample["gloss"],
                    "text": sample["text"],
                    "sign": sample["sign"],
                }
    return [samples[k] for k in samples]


@dataclass(frozen=True)
class Example:
    sequence: str
    signer: str
    sgn: torch.Tensor  # [T, F]
    gls: List[str]
    txt: List[str]


class ExamplesDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    @property
    def txt(self) -> List[List[str]]:
        return [e.txt for e in self.examples]

    @property
    def gls(self) -> List[List[str]]:
        return [e.gls for e in self.examples]

    @property
    def sequence(self) -> List[str]:
        return [e.sequence for e in self.examples]

    @property
    def signer(self) -> List[str]:
        return [e.signer for e in self.examples]


class _TorchBatch:
    def __init__(
        self,
        sequence: List[str],
        signer: List[str],
        sgn: Tuple[torch.Tensor, torch.Tensor],
        gls: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        txt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.sequence = sequence
        self.signer = signer
        self.sgn = sgn
        if gls is not None:
            self.gls = gls
        if txt is not None:
            self.txt = txt


def _tokenize_text(level: str) -> Callable[[str], List[str]]:
    if level == "char":
        return lambda s: list(s)
    return lambda s: s.split()


def _pad_1d(seqs: List[torch.Tensor], pad_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([int(s.size(0)) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out, lengths


def _pad_sgn(seqs: List[torch.Tensor], feature_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([int(s.size(0)) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) else 0
    out = torch.zeros((len(seqs), max_len, feature_size), dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, : s.size(0), :] = s
    return out, lengths


def make_data_iter(
    dataset: ExamplesDataset,
    batch_size: int,
    batch_type: str = "sentence",
    train: bool = False,
    shuffle: bool = False,
    gls_vocab: Optional[Vocabulary] = None,
    txt_vocab: Optional[Vocabulary] = None,
    level: str = "word",
    feature_size: int = 0,
) -> Iterable[_TorchBatch]:
    if batch_type != "sentence":
        raise NotImplementedError("native_data only supports batch_type='sentence' for now.")

    def collate(examples: List[Example]) -> _TorchBatch:
        sequences = [e.sequence for e in examples]
        signers = [e.signer for e in examples]

        sgn_batch, sgn_lengths = _pad_sgn([e.sgn for e in examples], feature_size)

        gls = None
        if gls_vocab is not None:
            gls_ids = [
                torch.tensor([gls_vocab.stoi[t] for t in e.gls], dtype=torch.long)
                for e in examples
            ]
            gls = _pad_1d(gls_ids, pad_value=gls_vocab.stoi[PAD_TOKEN])

        txt = None
        if txt_vocab is not None:
            txt_ids = []
            for e in examples:
                tokens = [BOS_TOKEN] + e.txt + [EOS_TOKEN]
                txt_ids.append(torch.tensor([txt_vocab.stoi[t] for t in tokens], dtype=torch.long))
            txt = _pad_1d(txt_ids, pad_value=txt_vocab.stoi[PAD_TOKEN])

        return _TorchBatch(sequence=sequences, signer=signers, sgn=(sgn_batch, sgn_lengths), gls=gls, txt=txt)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False,
        drop_last=False,
        collate_fn=collate,
    )
    return loader


def load_data(
    data_cfg: dict,
) -> Tuple[ExamplesDataset, ExamplesDataset, ExamplesDataset, Vocabulary, Vocabulary]:
    data_path = data_cfg.get("data_path", "./data")
    level = data_cfg["level"]
    txt_lowercase = data_cfg["txt_lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    if isinstance(data_cfg["train"], list):
        train_paths = [os.path.join(data_path, x) for x in data_cfg["train"]]
        dev_paths = [os.path.join(data_path, x) for x in data_cfg["dev"]]
        test_paths = [os.path.join(data_path, x) for x in data_cfg["test"]]
    else:
        train_paths = [os.path.join(data_path, data_cfg["train"])]
        dev_paths = [os.path.join(data_path, data_cfg["dev"])]
        test_paths = [os.path.join(data_path, data_cfg["test"])]

    tokenize_text = _tokenize_text(level)

    def normalize_txt(s: str) -> str:
        s = s.strip()
        return s.lower() if txt_lowercase else s

    def to_examples(paths: Sequence[str]) -> List[Example]:
        merged = _merge_samples(paths)
        out: List[Example] = []
        for s in merged:
            gls = tokenize_text(s["gloss"].strip())
            txt = tokenize_text(normalize_txt(s["text"]))
            # s["sign"] is shaped [F, T] (feature, time); filter by time steps
            time_len = int(s["sign"].size(1))
            if time_len <= max_sent_length and len(txt) <= max_sent_length:
                out.append(
                    Example(
                        sequence=s["name"],
                        signer=s["signer"],
                        sgn=(s["sign"] + 1e-8).transpose(0, 1).contiguous(),
                        gls=gls,
                        txt=txt,
                    )
                )
        return out

    train_examples = to_examples(train_paths)
    dev_examples = to_examples(dev_paths)
    test_examples = to_examples(test_paths)

    # random subset support (mirrors old torchtext split behavior)
    random_train_subset = data_cfg.get("random_train_subset", -1)
    if random_train_subset > -1 and random_train_subset < len(train_examples):
        train_examples = random.sample(train_examples, random_train_subset)

    random_dev_subset = data_cfg.get("random_dev_subset", -1)
    if random_dev_subset > -1 and random_dev_subset < len(dev_examples):
        dev_examples = random.sample(dev_examples, random_dev_subset)

    train_data = ExamplesDataset(train_examples)
    dev_data = ExamplesDataset(dev_examples)
    test_data = ExamplesDataset(test_examples)

    gls_vocab = build_vocab(
        field="gls",
        max_size=data_cfg.get("gls_voc_limit", 10000),
        min_freq=data_cfg.get("gls_voc_min_freq", 1),
        dataset=train_data,
        vocab_file=data_cfg.get("gls_vocab", None),
    )
    txt_vocab = build_vocab(
        field="txt",
        max_size=data_cfg.get("txt_voc_limit", 10000),
        min_freq=data_cfg.get("txt_voc_min_freq", 1),
        dataset=train_data,
        vocab_file=data_cfg.get("txt_vocab", None),
    )

    return train_data, dev_data, test_data, gls_vocab, txt_vocab
