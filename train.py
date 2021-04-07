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
        self.x0 = []
        self.t0 = []
        self.xn = []
        self.tn = []

    def collect(self, data: Record, loss, idx):
        free = lambda x: x.detach().cpu().numpy()
        self.x0.append(free(data.x0))
        self.t0.append(free(data.t0))
        self.xn.append(free(data.xn))
        self.tn.append(free(data.tn))
        self.loss_avg += (loss.item() - self.loss_avg) / idx

    def evaluate(self):
        x0 = np.concatenate(self.x0)
        t0 = np.concatenate(self.t0)
        xn = np.concatenate(self.xn)
        tn = np.concatenate(self.tn)

        mse = mean_squared_error(x0, xn)
        ev = explained_variance_score(x0, xn)
        bac = balanced_accuracy_score(t0, tn)

        return Score(loss=self.loss_avg, mse=mse, ev=ev, bac=bac)

    def clear(self):
        self.loss_avg = 0.
        self.predictions = []
        self.targets = []


def main():
    logging.info("loading data")
    cfg = Config
    data_bunch = get_data_bunch(cfg)

    logging.info("initializing model")
    model = MixedVAE(
        d_continuous=189,
        d_categories=16,
        d_embedding=128,
        depth=4,
    ).to(cfg.device)
    if cfg.resume:
        ckpt = torch.load(cfg.resume)
        model.load_state_dict(ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    metrics = Metrics()

    logging.info("starting training")
    best_score = 10e13
    result = Result()
    for epoch in range(1, 1000):
        model.train()
        metrics.clear()
        for idx, data in enumerate(data_bunch.train, start=1):
            data = Record(**data).to(cfg.device)
            for it in range(50):
                data, loss = model.train_step(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metrics.collect(data=data, loss=loss, idx=idx)
        result.train = metrics.evaluate()


        model.eval()
        metrics.clear()
        for idx, data in enumerate(data_bunch.dev, start=1):
            data = Record(**data).to(cfg.device)
            for it in range(100):
                xn, loss = model.eval_step(data)
                metrics.collect(data=data, loss=loss, idx=idx)
        result.dev = metrics.evaluate()
        logging.info(f"{epoch=:3d} {result.train} {result.dev}")

        if result.dev.loss < best_score:
            best_score = result.dev.loss
            result.best_epoch = epoch
            logging.info(f"saving checkpoint =====> {result.dev.loss=:.2f}")
            torch.save(model.state_dict(), f"ckpt.pt")

if __name__ == "__main__":
    main()
