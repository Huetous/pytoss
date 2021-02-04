import time
from functools import partial
from toss.utils import camel2snake
import torch
import matplotlib.pyplot as plt
import re
from toss.misc import AvgStats
from fastprogress.fastprogress import master_bar, progress_bar, format_time


class Callback:
    def __getattr__(self, k):
        return getattr(self.trainer, k)

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False

    @property
    def name(self):
        name = re.sub(r"Callback$", '', self.__class__.__name__)
        return camel2snake(name or "callback")

    def set_trainer(self, trainer):
        self.trainer = trainer


class TrainEvalCallback(Callback):
    ORDER = 1

    def begin_fit(self):
        self.trainer.epoch_number = 0.
        self.trainer.batch_number = 0

    def after_batch(self):
        if not self.in_train:
            return

        self.trainer.epoch_number += 1. / self.number_of_batches_in_epoch
        self.trainer.batch_number += 1

    def begin_epoch(self):
        self.trainer.epoch_number = self.epoch

        self.trainer.model.train()
        self.trainer.in_train = True

    def begin_validate(self):
        self.trainer.model.eval()
        self.trainer.in_train = False


class CudaCallback(Callback):
    ORDER = 2

    def begin_fit(self):
        self.model.cuda()

    def begin_batch(self):
        self.trainer.x_batch = self.x_batch.cuda()
        self.trainer.y_batch = self.y_batch.cuda()


class ParamScheduler(Callback):
    ORDER = 3

    def __init__(self, p_name, sched_funcs):
        self.p_name = p_name
        self.sched_funcs = sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.p_name] = f(self.epoch_number / self.epochs)

    def begin_batch(self):
        if self.in_train:
            self.set_param()


class LRFinder(Callback):
    ORDER = 3

    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.best_loss = 1e9
        self.best_lr = None

    def begin_batch(self):
        if not self.in_train:
            return

        pos = self.batch_number / self.max_iter
        self.lr = self.min_lr * (self.max_lr / self.min_lr) ** pos
        for pg in self.opt.param_groups:
            pg["lr"] = self.lr

    def after_step(self):
        if self.batch_number >= self.max_iter or self.loss >= self.best_loss * 10:
            raise CancelTrainException()
        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.best_lr = self.lr


class BatchTransformCallback(Callback):
    ORDER = 5

    def __init__(self, transform):
        self.transform = transform

    def begin_batch(self):
        self.trainer.x_batch = self.transform(self.trainer.x_batch)


class AvgStatsCallback(Callback):
    ORDER = 10

    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def begin_fit(self):
        metric_names = ["loss"] + [m.__name__ for m in self.train_stats.metrics]
        column_names = ["epoch"] + [f"train_{m}" for m in metric_names]
        column_names += [f"valid_{m}" for m in metric_names] + ["time"]
        self.logger(column_names)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.trainer)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


class Recorder(Callback):
    ORDER = 11

    def begin_fit(self):
        self.losses = []
        self.lrs = [[] for _ in self.opt.param_groups]

    def after_batch(self):
        if not self.in_train:
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_loss(self, skip_last=0):
        plt.plot(self.losses[:len(self.losses) - skip_last])
        plt.ylabel("Loss")
        plt.xlabel("Number of batch")
        plt.show()

    def plot_lr(self, pg_id=-1):
        plt.plot(self.lrs[pg_id])
        plt.ylabel("Lr")
        plt.xlabel("Number of batch")
        plt.show()

    def plot(self, skip_last=0, pg_id=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pg_id]
        n = len(losses) - skip_last
        plt.xscale("log")
        plt.plot(lrs[:n], losses[:n])
        plt.show()


class ProgressCallback(Callback):
    ORDER = -1

    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.trainer.logger = partial(self.mbar.write, table=True)

    def after_fit(self):
        self.mbar.on_iter_end()

    def after_batch(self):
        self.pb.update(self.iter)

    def begin_epoch(self):
        self.set_pb()

    def begin_validate(self):
        self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.trainer.data, parent=self.mbar)
        self.mbar.update(self.epoch)


class CancelTrainException(Exception):
    pass


class CancelEpochException(Exception):
    pass


class CancelBatchException(Exception):
    pass
