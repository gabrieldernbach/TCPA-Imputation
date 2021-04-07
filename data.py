import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass, asdict


class MixedMissingEntryDataset(Dataset):
    def __init__(self, x, t, min_missing, max_missing):
        self.x = x.detach().float()
        self.t = t.detach().long()
        #  in an unfortunate event, self.tumortype.max()
        #  could return less classes as there are across the splits
        #  when the specific split is indeed lacking a class.
        #  This should be protected by a stratified split.
        self.n_classes = self.t.max() + 1
        self.min_missing = min_missing
        self.max_missing = max_missing

    def __getitem__(self, idx):
        # get original `x0` and imputed `xn` proteins
        x0 = self.x[idx]
        prob = np.random.uniform(self.min_missing, self.max_missing)
        xmask = np.random.binomial(1, prob, x0.shape)
        xmask = torch.tensor(xmask, dtype=torch.bool).detach()
        xn = x0.clone()
        xn[xmask] = torch.randn(xmask.shape, dtype=torch.float)[xmask]

        # get original `t0` and imputed `tn` tumortypes
        t0 = self.t[idx]
        tmask = (torch.rand(1) > 0.5).detach()
        tn = t0.clone()
        if tmask:
            tn = torch.randint(0, self.n_classes, size=(1,))
        tne = torch.tensor([])

        record = Record(x0=x0, xn=xn, xmask=xmask, t0=t0, tn=tn, tne=tne, tmask=tmask)
        return asdict(record)

    def __len__(self):
        return len(self.x)


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


def split(df, col="target", frac=0.60, seed=0):
    """stratified split along col='target'"""
    grouped = df.groupby(col, group_keys=False)
    collect = []
    for name, group in grouped:
        inside = group.sample(frac=frac, random_state=seed)
        outside = group.drop(inside.index)
        collect.append((inside, outside))
    ins, outs = [pd.concat(f) for f in zip(*collect)]
    return ins, outs


def add_cv_col(df):
    df["cv"] = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for idx, (_, test_idx) in enumerate(skf.split(df, df.target)):
        df.loc[test_idx, "cv"] = idx
    return df


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
        num_workers=cfg.num_worker,
        shuffle=True,
        drop_last=False,
    )
    return dataloader


def dataframe2databunch(df, cfg):
    df = add_cv_col(df)
    test = df[df.cv == cfg.seed].drop(columns="cv")
    rest = df[df.cv != cfg.seed].drop(columns="cv")
    train, dev = split(rest, frac=0.8, seed=cfg.seed)

    train_proteins = np.array(train.loc[:, train.columns != "target"])
    train_tumortype = np.array(train.loc[:, train.columns == "target"])
    dev_proteins = np.array(dev.loc[:, dev.columns != "target"])
    dev_tumortype = np.array(dev.loc[:, dev.columns == "target"])
    test_proteins = np.array(test.loc[:, test.columns != "target"])
    test_tumortype = np.array(test.loc[:, test.columns == "target"])

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
    df = pd.read_csv("TCPA_data_sel.csv")
    df["target"] = df.Tumor.astype("category").cat.codes
    df = df.drop(columns=["ID", "Tumor"])
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

    def to(self, device):
        self.x0 = self.x0.to(device)
        self.xn = self.xn.to(device)
        self.t0 = self.t0.to(device)
        self.tn = self.tn.to(device)
        self.tne = self.tne.to(device)
        self.xmask = self.xmask.to(device)
        self.tmask = self.tmask.to(device)
        return self


@dataclass(frozen=True)
class DataBunch:
    train: DataLoader
    dev: DataLoader
    test: DataLoader
    normalizer: Normalizer


@dataclass(frozen=True)
class Config:
    seed: int = 0
    batch_size: int = 4096
    num_worker: int = 32
    epochs: int = 300
    resume: str = "ckpt.pt" #None # "ckpt.pt"
    device: str = "cpu"
    mixed_precision: bool = True


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
