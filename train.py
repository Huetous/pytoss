from toss.utils import listify
import torch
from functools import partial
from toss.callbacks import TrainEvalCallback, CancelTrainException, CancelEpochException, CancelBatchException


class Trainer:
    CALLBACK_NAMES = ["begin_fit", "begin_epoch", "begin_batch", "begin_validate",
                      "after_fit", "after_epoch", "after_batch",
                      "after_loss", "after_pred", "after_backward", "after_step",
                      "after_cancel_train", "after_cancel_epoch", "after_cancel_batch"]

    def __init__(self, model, data, loss_func, opt, cbs=None):
        self.model = model
        self.data = data
        self.loss_func = loss_func
        self.opt = opt
        self.in_train = False

        self.cbs = [TrainEvalCallback()]
        for cb in listify(cbs):
            if isinstance(cb, partial):
                cb = cb()
            setattr(self, cb.name, cb)
            self.cbs.append(cb)
        self.cbs = list(sorted(self.cbs, key=lambda x: x.ORDER))

    def __call__(self, cb_name):
        res = False
        for cb in self.cbs:
            res = cb(cb_name) or res
        return res

    def fit(self, epochs):
        self.epochs = epochs
        self.loss = torch.tensor(0.)
        try:
            for cb in self.cbs:
                cb.set_trainer(self)

            self("begin_fit")
            for epoch in range(epochs):
                self.epoch = epoch
                self("begin_epoch")
                self._all_batches(self.data.train_gen)

                with torch.no_grad():
                    self("begin_validate")
                    self._all_batches(self.data.valid_gen)

                self("after_epoch")

        except CancelTrainException:
            self("after_cancel_train")
        finally:
            self("after_fit")

    def _all_batches(self, gen):
        self.number_of_batches_in_epoch = len(gen)
        try:
            for xb, yb in gen:
                self._batch(xb, yb)

        except CancelEpochException:
            self("after_cancel_epoch")

    def _batch(self, x, y):
        try:
            self.x_batch = x
            self.y_batch = y
            self("begin_batch")

            self.pred = self.model(self.x_batch)
            self("after_pred")

            self.loss = self.loss_func(self.pred, self.y_batch)
            self("after_loss")

            if self.in_train:
                self.loss.backward()
                self("after_backward")

                self.opt.step()
                self("after_step")
                self.opt.zero_grad()

        except CancelBatchException:
            self("after_cancel_batch")
        finally:
            self("after_batch")

    def help(self):
        print("Available callbacks: ", self.CALLBACK_NAMES)

