import numpy as np
from torch import multiprocessing
import time
import os

DATA = None
COLLATE = None

# If any thread takes longer than this, terminate it.
TIMEOUT = 60.0 * 5

# def _process(batch, return_list):
#     return COLLATE([DATA[x] for x in batch])


def log(txt):
    os.system('echo "' + txt + '" >> log.txt')
    print(txt)


def _process2(batch_of_batches, return_list, id):
    for batch in batch_of_batches:
        try:
            return_list.append(COLLATE([DATA[x] for x in batch]))
            # print("WORKED", id, batch)
        except:
            print("FAILED", id, batch)


def _collate_none(x):
    return x[0]


class MultiLoader(object):
    def __init__(
        self,
        data,
        num_dataloader_workers=1,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate_none,
        max_voxels_in_memory=80,
    ):
        self.data = data
        self.num_dataloader_workers = num_dataloader_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.max_voxels_in_memory = max_voxels_in_memory

        # JDD added below based on
        # https://github.com/pytorch/pytorch/issues/67844 See also
        # https://pytorch.org/docs/stable/multiprocessing.html#multiprocessing-cuda-sharing-details
        # JDD NO: multiprocessing.set_sharing_strategy("file_system")

    def __len__(self) -> int:
        """Returns number of batches."""
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        else:
            return (len(self.data) // self.batch_size) + 1

    def _add_procs(self, id):
        global TIMEOUT

        if len(self.return_list) >= self.max_voxels_in_memory:
            # You already have enough
            return

        cur_time = time.time()

        # Go through existing procs and kill those that have been running
        # for too long, and join those that have finished.
        for i, (p, timestamp) in enumerate(self.procs):
            if p.is_alive():
                if cur_time - timestamp > TIMEOUT:
                    # It's been running for too long
                    print("timed out, killing a process: " + p.name)
                    p.terminate()
                    self.procs[i] = None
            else:
                # It's finished with the calculation
                # print("finished, joining a process: " + p.name + " (" + str(cur_time - timestamp) + ")")
                p.join()
                self.procs[i] = None

        # Remove the Nones
        self.procs = [p for p in self.procs if p is not None]

        # Keep adding new procs until you reach the limit
        while (
            len(self.procs) < self.num_dataloader_workers
            and len(self.batches_of_batches) > 0
        ):
            batch = self.batches_of_batches.pop(0)

            p = multiprocessing.Process(
                target=_process2,
                args=(batch, self.return_list, id),
                name=("process_" + str(self.name_idx + 1)),
            )
            self.name_idx = self.name_idx + 1
            p.start()
            self.procs.append((p, cur_time))

    def __iter__(self):
        global DATA
        global COLLATE

        DATA = self.data
        COLLATE = self.collate_fn

        # id is for debugging. Not critical.
        id = str(time.time())

        # Avoiding multiprocessing.Pool because I want to terminate threads if
        # they take too long.

        # Just indexes to the batches. For shuffling.
        data_idxs = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(data_idxs)
        batches_idxs = []
        num_data = len(data_idxs)
        for i in range(0, num_data, self.batch_size):
            batches_idxs.append(data_idxs[i : i + self.batch_size])

        # Batch the batches. Each of these uber batches goes to its own
        # processor.
        batches_to_process_per_proc = (
            self.max_voxels_in_memory // self.num_dataloader_workers
        )
        batches_to_process_per_proc = (
            1 if batches_to_process_per_proc == 0 else batches_to_process_per_proc
        )

        self.batches_of_batches = []
        for j in range(1 + len(batches_idxs) // batches_to_process_per_proc):
            batch_of_batches = batches_idxs[
                j * batches_to_process_per_proc : (j + 1) * batches_to_process_per_proc
            ]
            if len(batch_of_batches) > 0:
                self.batches_of_batches.append(batch_of_batches)

        # For debugging...
        # should be "file_system"
        # print(multiprocessing.get_sharing_strategy())
        # print(len(self.data))
        # print(batches_idxs[-1])
        # print(self.batches_of_batches[-1])
        # print("-----")

        self.procs = []
        manager = multiprocessing.Manager()
        self.return_list = manager.list()

        self.name_idx = 0

        # Let's give the grid-generation a little head start (decided not to do
        # this).
        # self._add_procs()
        # time.sleep(15)

        count = 0
        while len(self.batches_of_batches) > 0:
            self._add_procs(id)

            # Wait until you've got at least one ready
            while len(self.return_list) == 0:
                # print("Waiting for a voxel grid to finish... you might try increasing --max_voxels_in_memory")
                time.sleep(0.1)

            # Yield the data as it is needed
            while count < num_data:  # len(self.return_list) > 0 or
                if len(self.return_list) == 0:
                    time.sleep(0.1)
                    continue

                item = self.return_list.pop(0)

                if len(self.return_list) < self.max_voxels_in_memory * 0.1:
                    # Getting low on voxels...
                    self._add_procs(id)

                count = count + 1
                # print(count)

                yield item

        # ===== WORKS BUT IF ERROR ON ANY THREAD, HANGS WHOLE PROGRAM ====
        # with multiprocessing.Pool(self.num_dataloader_workers) as p:
        #     items = p.imap_unordered(_process, these_batches_idxs)
        #     for item in items:
        #         import pdb; pdb.set_trace()
        #         yield item

        #     for item in p.imap_unordered(_process, these_batches_idxs):
        #         print(item)
        #         yield item

        # ===== HARRISON ORIGINAL IMPLEMENTATION =====
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

        # ===== For debugging =====
        # Very slow, but avoids multiprocessing. Also, I think it fixes the
        # problem with run-away open files problem.
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
