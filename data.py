import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass, asdict


class MissingEntryDataset(Dataset):
    def __init__(self, data, min_missing, max_missing):
        self.data = data.detach().float()
        self.min_missing = min_missing
        self.max_missing = max_missing

    def __getitem__(self, idx):
        x0 = self.data[idx]

        prob = np.random.uniform(self.min_missing, self.max_missing)
        mask = np.random.binomial(1, prob, x0.shape)
        mask = torch.tensor(mask, dtype=torch.bool)

        xn = x0.clone()
        xn[mask] = torch.randn(mask.shape, dtype=torch.float)[mask]
        return x0, xn, mask

    def __len__(self):
        return len(self.data)


class Normalizer:
    def __init__(self):
        self.mean = np.array
        self.std = np.array

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        data -= self.mean
        data /= self.std
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def split(df, frac, seed):
    inside = df.sample(frac=frac, random_state=seed)
    outside = df.drop(inside.index)
    return inside, outside


def wrap_loader(data, cfg):
    data_tensor = torch.tensor(data)
    dataset = MissingEntryDataset(
        data_tensor,
        min_missing=0.1,
        max_missing=0.9
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        num_workers=0,
        shuffle=True
    )
    return dataloader


def dataframe2databunch(df, cfg):
    n, d = df.shape
    train, rest = split(df, frac=0.7, seed=cfg.seed)
    dev, test = split(rest, frac=0.5, seed=cfg.seed)
    train, dev, test = [np.array(x) for x in [train, dev, test]]

    normalizer = Normalizer()
    train = normalizer.fit_transform(train)
    dev = normalizer.transform(dev)
    test = normalizer.transform(test)

    train_dl = wrap_loader(train, cfg)
    dev_dl = wrap_loader(dev, cfg)
    test_dl = wrap_loader(test, cfg)
    data_bunch = DataBunch(
        train=train_dl,
        dev=dev_dl,
        test=test_dl,
        normalizer=normalizer,
        n_sample=n,
        d_dimension=d,
    )
    return data_bunch


def task_tcpa():
    df = pd.read_csv("TCPA_data_sel.csv")
    # onehot = pd.get_dummies(df.Tumor, prefix="Tumor")
    # df = pd.concat([df, onehot], axis=1)
    df = df.drop(columns=["ID", "Tumor"])
    return df


def task_dag():
    # todo:
    #  forward adjacency?
    #  make this configurable?
    adjacency, samples = torch.load("dag_nn.pt")
    df = pd.DataFrame(samples)
    return df


def get_data_bunch(cfg):
    df = {
        "tcpa": task_tcpa,
        "dag": task_dag,
    }[cfg.task]()
    data_bunch = dataframe2databunch(df, cfg)
    return data_bunch


@dataclass(frozen=True)
class DataBunch:
    train: DataLoader
    dev: DataLoader
    test: DataLoader
    normalizer: Normalizer
    n_sample: int
    d_dimension: int


@dataclass(frozen=True)
class Config:
    task: str = "tcpa"
    seed: int = 0
    batch_size: int = 64
    epochs: int = 300
    resume: str = None #"ckpt.pt"

@dataclass(frozen=True)
class Score:
    loss: float
    mse: float
    r2: float

    def __str__(self):
        return f"{self.loss=:.13f}, {self.mse=:.13f}, {self.r2=:.4f}"

@dataclass(frozen=False)
class Result:
    train: Optional[Score] = None
    dev: Optional[Score] = None
    test: Optional[Score] = None

    def __iter__(self):
        """a generator of the score dictionaries with mode=[train, dev, test]"""
        return (dict(mode=k, **v) for k, v in asdict(self).items())
