import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass, asdict


class MixedMissingEntryDataset(Dataset):
    def __init__(self, proteins, tumortype, min_missing, max_missing):
        self.proteins = proteins.detach().float()
        self.tumortype = tumortype.detach().long()
        #  in an unfortunate event, self.tumortype.max()
        #  could return less classes as there are in the total dataset,
        #  when the subset is indeed lacking a class.
        #  This should be protected by a stratified split.
        self.n_classes = self.tumortype.max() + 1
        self.min_missing = min_missing
        self.max_missing = max_missing

    def __getitem__(self, idx):
        # get original `x0` and imputed `xn` proteins
        x0 = self.proteins[idx]
        prob = np.random.uniform(self.min_missing, self.max_missing)
        xmask = np.random.binomial(1, prob, x0.shape)
        xmask = torch.tensor(xmask, dtype=torch.bool).detach()
        xn = x0.clone()
        xn[xmask] = torch.randn(xmask.shape, dtype=torch.float)[xmask]

        # get original `t0` and imputed `tn` tumortypes
        t0 = self.tumortype[idx]
        tmask = (torch.rand(1) > 0.5).detach()
        tn = t0.clone()
        if tmask:
            tn = torch.randint(0, self.n_classes, size=(1,))

        record = Record(x0=x0, xn=xn, xmask=xmask, t0=t0, tn=tn, tne=[], tmask=tmask)
        return asdict(record)

    def __len__(self):
        return len(self.proteins)


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


def wrap_loader(proteins, tumortype, cfg):
    proteins = torch.tensor(proteins)
    tumortype = torch.tensor(tumortype)
    dataset = MixedMissingEntryDataset(
        proteins,
        tumortype,
        min_missing=0.1,
        max_missing=0.9,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        num_workers=0,
        shuffle=True,
    )
    return dataloader


def dataframe2databunch(df, cfg):
    train, rest = split(df, frac=0.7, seed=cfg.seed)
    dev, test = split(rest, frac=0.5, seed=cfg.seed)

    train_proteins = np.array(train.loc[:, train.columns != "Tumor"])
    train_tumortype = np.array(train.loc[:, train.columns == "Tumor"])
    dev_proteins = np.array(dev.loc[:, dev.columns != "Tumor"])
    dev_tumortype = np.array(dev.loc[:, dev.columns == "Tumor"])
    test_proteins = np.array(test.loc[:, test.columns != "Tumor"])
    test_tumortype = np.array(test.loc[:, test.columns == "Tumor"])

    normalizer = Normalizer()
    train_proteins = normalizer.fit_transform(train_proteins)
    dev_proteins = normalizer.transform(dev_proteins)
    test_proteins = normalizer.transform(test_proteins)

    train_dl = wrap_loader(train_proteins, train_tumortype, cfg)
    dev_dl = wrap_loader(dev_proteins, dev_tumortype, cfg)
    test_dl = wrap_loader(test_proteins, test_tumortype, cfg)
    data_bunch = DataBunch(
        train=train_dl,
        dev=dev_dl,
        test=test_dl,
        normalizer=normalizer,
    )
    return data_bunch


def task_tcpa():
    # ideally this would have one mode including the tumor_type
    # and another mode with only the protein data
    df = pd.read_csv("TCPA_data_sel.csv")
    # onehot = pd.get_dummies(df.Tumor, prefix="Tumor")
    # df = pd.concat([df, onehot], axis=1)
    # df = df.drop(columns=["ID", "Tumor"])
    df["Tumor"] = df.Tumor.astype("category")
    df["Tumor"] = df.Tumor.cat.codes
    df = df.drop(columns=["ID"])
    return df


def get_data_bunch(cfg):
    df = task_tcpa()
    data_bunch = dataframe2databunch(df, cfg)
    return data_bunch


@dataclass
class Record:
    """
    A data record storing continous x and categorical t values,
    A suffix 0 indicates original data, n an instance with missing values
    Entries where xmask==0 are identical between x0 and xn,
    Entries where xmask==1 have been set with a random value in xn
    """
    x0: torch.tensor  # continuous original
    xn: torch.tensor  # continuous imputed
    t0: torch.tensor  # categorical original
    tn: torch.tensor  # categorical imputed
    tne: torch.tensor  # embedded categories
    xmask: torch.tensor  # mask of which xn are imputed
    tmask: torch.tensor  # mask of which tn are imputed


@dataclass(frozen=True)
class DataBunch:
    train: DataLoader
    dev: DataLoader
    test: DataLoader
    normalizer: Normalizer


@dataclass(frozen=True)
class Config:
    seed: int = 0
    batch_size: int = 64
    epochs: int = 300
    resume: str = None # "ckpt.pt"


@dataclass(frozen=True)
class Score:
    loss: float
    mse: float
    ev: float
    bac: float

    def __str__(self):
        return f"{self.loss=:.6f}, {self.mse=:.6f}, {self.ev=:.3f}, {self.bac=:.3f}"


@dataclass(frozen=False)
class Result:
    train: Optional[Score] = None
    dev: Optional[Score] = None
    test: Optional[Score] = None

    def __iter__(self):
        """a generator of the score dictionaries with mode=[train, dev, test]"""
        return (dict(mode=k, **v) for k, v in asdict(self).items())
