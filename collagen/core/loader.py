import numpy as np
from torch import multiprocessing
import copy
import time

DATA = None
COLLATE = None


def _process(batch, return_list):
    return COLLATE([DATA[x] for x in batch])

def _process2(batch_of_batches, return_list):
    for batch in batch_of_batches:
        return_list.append(COLLATE([DATA[x] for x in batch]))
    # return COLLATE([DATA[x] for x in batch])


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

        # JDD added below based on
        # https://github.com/pytorch/pytorch/issues/67844 See also
        # https://pytorch.org/docs/stable/multiprocessing.html#multiprocessing-cuda-sharing-details
        multiprocessing.set_sharing_strategy("file_system")

    def __len__(self):
        # So it's returning number of batches, not number of examples (though
        # same if batch size is 1)
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        else:
            return (len(self.data) // self.batch_size) + 1

    def __iter__(self):
        global DATA
        global COLLATE

        DATA = self.data
        COLLATE = self.collate_fn

        # Just indexes to the batches. For shuffling.
        data_idxs = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(data_idxs)
        batches_idxs = []
        for i in range(0, len(data_idxs), self.batch_size):
            batches_idxs.append(data_idxs[i : i + self.batch_size])

        # should be "file_system"
        # print(multiprocessing.get_sharing_strategy())  

        manager = multiprocessing.Manager()

        batches_of_batches_fac = 10
        while len(batches_idxs) > 0:
            these_batches_idxs = batches_idxs[:self.num_dataloader_workers * batches_of_batches_fac]
            batches_idxs = batches_idxs[self.num_dataloader_workers * batches_of_batches_fac:]
            
            batches_of_batches = []
            for j in range(self.num_dataloader_workers):
                batches_of_batches.append(
                    these_batches_idxs[j * batches_of_batches_fac: (j + 1) * batches_of_batches_fac]
                )

            # Avoiding multiprocessing.Pool because I want to terminate threads
            # if they take too long.
            return_list = manager.list()
            procs = []
            bool_list = []
            for i, batche_of_batches in enumerate(batches_of_batches):
                p = multiprocessing.Process(
                    target = _process2, 
                    args = (batche_of_batches, return_list),
                    name = ('process_' + str(i+1))
                )
                procs.append(p)
                bool_list.append(True)
                p.start()
                # print('starting', p.name)

            TIMEOUT = 60.0

            start = time.time()
            while time.time() - start <= TIMEOUT:
                for i in range(self.num_dataloader_workers):
                    bool_list[i] = procs[i].is_alive()
                if np.any(bool_list):  
                    time.sleep(.1)  
                else:
                    break
            else:
                print("timed out, killing all processes")
                for p in procs:
                    p.terminate()

            for p in procs:
                # print('stopping', p.name,'=', p.is_alive())
                p.join()

            for item in return_list:
                yield item


            # with multiprocessing.Pool(self.num_dataloader_workers) as p:
            #     items = p.imap_unordered(_process, these_batches_idxs)
            #     for item in items:
            #         import pdb; pdb.set_trace()
            #         yield item

                # for item in p.imap_unordered(_process, these_batches_idxs):
                #     print(item)
                #     yield item

        # This serves the data.
        # with multiprocessing.Pool(self.num_dataloader_workers) as p:
        #     for item in p.imap_unordered(_process, batches_idxs):
        #         # JDD: Note the need to make a deep copy here:
        #         # https://github.com/pytorch/pytorch/issues/11201
        #         # item_cp = copy.deepcopy(item)
        #         # del item
        #         # import pdb; pdb.set_trace()
        #         # item_cp = np.asarray(item, dtype=object)
        #         # del item
        #         yield item

        # For debugging. Very slow, but avoids multiprocessing. Also, fixes
        # run-away open files problem.
        # for idxs in batches_idxs:
        #     item = _process(idxs)
        #     yield item

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
