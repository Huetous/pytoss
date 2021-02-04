import torch


class DataSet:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.n = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.batch_size):
            yield self.dataset[idxs[i: i + self.batch_size]]


class DataBunch:
    def __init__(self, train_ds, valid_ds, batch_size, n_in=None, n_out=None):
        self.train_gen = DataLoader(train_ds, batch_size, shuffle=True)
        self.valid_gen = DataLoader(valid_ds, batch_size)
        self.n_in = n_in
        self.n_out = n_out

    @property
    def train_ds(self):
        return self.train_gen.dataset

    @property
    def valid_ds(self):
        return self.valid_gen.dataset
