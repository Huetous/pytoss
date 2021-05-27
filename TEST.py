import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import SGD

from toss.data import DataSet, DataBunch
from toss.callbacks import BatchTransformCallback, CudaCallback, AvgStatsCallback, Recorder, ParamScheduler, LRFinder
from toss.metrics import accuracy
from toss.misc import get_mnist, normalize, mnist_resize, get_cnn_model, conv_layer, append_stats
from toss.hooks import Hooks, plot_hooks, plot_hooks_act, plot_hooks_min_act
from toss.train import Trainer
from toss.optimizers import Optimizer
from toss.scheduling import combine_scheds, sched_cos
from toss.misc import init_cnn

x_train, y_train, x_valid, y_valid = get_mnist()
x_train, x_valid = normalize(x_train, x_valid)

BATCH_SIZE = 512
N_CLASSES = y_train.max() + 1
LOSS_FUNC = F.cross_entropy

train_ds = DataSet(x_train, y_train)
valid_ds = DataSet(x_valid, y_valid)

data = DataBunch(train_ds, valid_ds, batch_size=BATCH_SIZE, n_classes=N_CLASSES)

nfs = [8, 16, 32, 32]
sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])
cbfs = [Recorder(),
        partial(AvgStatsCallback, accuracy),
        CudaCallback(),
        partial(ParamScheduler, "lr", sched),
        LRFinder(),
        partial(BatchTransformCallback, mnist_resize)]

model = get_cnn_model(data, nfs, conv_layer,
                      upper_bound=6.,
                      leak=0.1,
                      sub=0.4
                      )
init_cnn(model)
# steppers =
# opt = Optimizer(model.parameters(), steppers=steppers)
opt = SGD(model.parameters(), lr=0.9)
trainer = Trainer(model, data, LOSS_FUNC, opt=opt, cbs=cbfs)
trainer.fit(1)
trainer.recorder.plot(skip_last=5)
trainer.recorder.plot_lr()
print(trainer.lr_finder.best_lr)


