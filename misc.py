import gzip
import pickle
from torch import tensor
from fastai import datasets
from toss.utils import listify


def append_stats(hook, mod, inp, out):
    if not hasattr(hook, "stats"):
        hook.stats = ([], [], [])
    means, stds, hists = hook.stats
    means.append(out.data.mean().cpu())
    stds.append(out.data.std().cpu())
    hists.append(out.data.cpu().histc(40, -10, 10))


def get_mnist():
    MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"
    path = datasets.download_data(MNIST_URL, ext=".gz")
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return map(tensor, (x_train, y_train, x_valid, y_valid))


def normalize(train, valid):
    m = train.mean()
    s = train.std()
    train -= m
    train /= s
    valid -= m
    valid /= s
    return train, valid


class AvgStats:
    def __init__(self, metrics, in_train):
        self.metrics = listify(metrics)
        self.in_train = in_train

        self.total_loss = 0
        self.total_size = 0
        self.total_metrics = [0.] * len(self.metrics)

    def __repr__(self):
        if not self.total_size:
            return ""
        return f"{'train' if self.in_train else 'valid'} : {self.avg_stats}"

    @property
    def all_stats(self):
        return [self.total_loss.item()] + self.total_metrics

    @property
    def avg_stats(self):
        return [o / self.total_size for o in self.all_stats]

    def reset(self):
        self.total_loss = 0
        self.total_size = 0
        self.total_metrics = [0.] * len(self.metrics)

    def accumulate(self, trainer):
        batch_size = trainer.x_batch.shape[0]
        self.total_loss += trainer.loss * batch_size
        self.total_size += batch_size
        for i, metric in enumerate(self.metrics):
            self.total_metrics[i] += metric(trainer.pred, trainer.y_batch) * batch_size
