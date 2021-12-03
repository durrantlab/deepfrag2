import numpy as np
from torch import multiprocessing


DATA = None
COLLATE = None


def _process(batch):
    return COLLATE([DATA[x] for x in batch])


def _collate_none(x):
    return x[0]


class MultiLoader(object):
    def __init__(
        self, data, num_dataloader_workers=1, batch_size=1, shuffle=False, collate_fn=_collate_none
    ):
        self.data = data
        self.num_dataloader_workers = num_dataloader_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

    def __len__(self):
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        else:
            return (len(self.data) // self.batch_size) + 1

    def __iter__(self):
        global DATA
        global COLLATE

        DATA = self.data
        COLLATE = self.collate_fn

        work = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(work)
        batches = []
        for i in range(0, len(work), self.batch_size):
            batches.append(work[i : i + self.batch_size])

        with multiprocessing.Pool(self.num_dataloader_workers) as p:
            for item in p.imap_unordered(_process, batches):
                yield item

    def batch(self, batch_size):
        return DataBatch(self, batch_size)

    def map(self, fn):
        return DataLambda(self, fn)


class DataLambda(MultiLoader):
    def __init__(self, data, fn):
        self.data = data
        self.fn = fn

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for item in self.data:
            yield self.fn(item)


class DataBatch(MultiLoader):
    def __init__(self, data, batch: int):
        self.data = data
        self.batch = batch

    def __len__(self):
        if len(self.data) % self.batch == 0:
            return len(self.data) // self.batch
        else:
            return (len(self.data) // self.batch) + 1

    def __iter__(self):
        n = 0
        batch = []
        for item in self.data:
            n += 1
            batch.append(item)

            if n == self.batch:
                yield batch
                batch = []
                n = 0

        if n != 0:
            yield batch
