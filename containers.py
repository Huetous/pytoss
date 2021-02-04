from toss.utils import listify


class ListContainer:
    def __init__(self, items):
        self.items = listify(items)

    def __getitem__(self, indexes):
        if isinstance(indexes, (int, slice)):
            return self.items[indexes]
        if isinstance(indexes[0], bool):
            assert len(indexes) == len(self)
            return [o for m, o in zip(indexes, self.items) if m]
        return [self.items[i] for i in indexes]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del (self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items) \n {self.items[:10]}'
        if len(self) > 10:
            res += " ...]"
        return res
