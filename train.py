import torch

from data import Config, Result, Score, get_data_bunch
from model import VAE
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d:%H:%M:%S',
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
)


class Metrics:
    """A callback observing training and evaluation."""

    def __init__(self):
        self.mse = 0.
        self.r2 = 0.
        self.loss_avg = 0.

        self.predictions = []
        self.targets = []

    def collect(self, prediction, target, loss, idx):
        self.predictions.append(prediction.detach().cpu().numpy())
        self.targets.append(target.detach().cpu().numpy())
        self.loss_avg += (loss.item() - self.loss_avg) / idx

    def evaluate(self):
        predictions = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)

        self.mse = mean_squared_error(targets, predictions)
        self.r2 = r2_score(targets, predictions)
        return Score(loss=self.loss_avg, mse=self.mse, r2=self.r2)

    def clear(self):
        self.loss_avg = 0.
        self.predictions = []
        self.targets = []


def main():
    cfg = Config
    data_bunch = get_data_bunch(cfg)

    model = VAE(ins=data_bunch.d_dimension)
    if cfg.resume:
        ckpt = torch.load(cfg.resume)
        model.load_state_dict(ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    metrics = Metrics()

    result = Result()
    for epoch in range(1, 100):
        model.train()
        loss_monitor = 0
        metrics.clear()
        for idx, (x0, xn, missing_mask) in enumerate(data_bunch.train, start=1):
            for it in range(epoch):
                xn, loss = model.train_step(x0, xn, missing_mask)
                loss_monitor = 0.9 * loss_monitor + 0.1 * loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metrics.collect(prediction=xn, target=x0, loss=loss, idx=idx)
            (idx % 10 == 0) and logging.info(f"{loss_monitor=:.13f}")

        result.train = metrics.evaluate()
        torch.save(model.state_dict(), "ckpt.pt")

        model.eval()
        metrics.clear()
        for idx, (x0, xn, missing_mask) in enumerate(data_bunch.dev, start=1):
            for it in range(epoch):
                xn, loss = model.eval_step(x0, xn, missing_mask)
            metrics.collect(prediction=xn, target=x0, loss=loss, idx=idx)
        result.dev = metrics.evaluate()
        logging.info(f"{epoch=:3d} {result.train} {result.dev}")


if __name__ == "__main__":
    main()
