import logging

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, explained_variance_score, balanced_accuracy_score

from data import Config, Result, Score, Record, get_data_bunch
from model import MixedVAE

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d:%H:%M:%S',
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
)


class Metrics:
    """A callback observing training and evaluation."""

    def __init__(self):
        self.loss_avg = 0.
        self.xn = []
        self.x0 = []
        self.t0 = []
        self.tn = []

    def collect(self, data: Record, loss, idx):
        free = lambda x: x.detach().cpu().numpy()
        self.xn.append(free(data.xn))
        self.x0.append(free(data.x0))
        self.tn.append(free(data.tn))
        self.t0.append(free(data.t0))
        self.loss_avg += (loss.item() - self.loss_avg) / idx

    def evaluate(self):
        xn = np.concatenate(self.xn)
        x0 = np.concatenate(self.x0)
        tn = np.concatenate(self.tn)
        t0 = np.concatenate(self.t0)

        mse = mean_squared_error(x0, xn)
        ev = explained_variance_score(x0, xn)
        bac = balanced_accuracy_score(t0, tn)

        return Score(loss=self.loss_avg, mse=mse, ev=ev, bac=bac)

    def clear(self):
        self.loss_avg = 0.
        self.predictions = []
        self.targets = []


def main():
    cfg = Config
    data_bunch = get_data_bunch(cfg)

    model = MixedVAE(
        d_continuous=189,
        d_categories=16,
        d_embedding=128,
        depth=4,
    )
    if cfg.resume:
        ckpt = torch.load(cfg.resume)
        model.load_state_dict(ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    metrics = Metrics()

    result = Result()
    for epoch in range(1, 100):
        model.train()
        metrics.clear()
        for idx, data in enumerate(data_bunch.train, start=1):
            data = Record(**data)
            for it in range(epoch):
                data, loss = model.train_step(data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metrics.collect(data=data, loss=loss, idx=idx)

        result.train = metrics.evaluate()
        torch.save(model.state_dict(), "ckpt.pt")

        model.eval()
        metrics.clear()
        for idx, data in enumerate(data_bunch.dev, start=1):
            data = Record(**data)
            for it in range(epoch):
                xn, loss = model.eval_step(data)
            metrics.collect(data=data, loss=loss, idx=idx)
        result.dev = metrics.evaluate()
        logging.info(f"{epoch=:3d} {result.train} {result.dev}")


if __name__ == "__main__":
    main()
