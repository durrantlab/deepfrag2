"""Contains the MultiLoader class, which serves up the data for
training and testing.
"""

from typing import Any, List
import numpy as np
from torch import multiprocessing
import time
import os
import traceback
from datetime import datetime
import threading
import platform

DATA = None
COLLATE = None

# If any thread takes longer than this, terminate it.
TIMEOUT = 60.0 * 5


def log(txt: str):
    """Log a message to the log file and print it to the console.
    
    Args:
        txt (str): The message to log.
    """
    os.system(f'echo "{txt}" >> log.txt')
    print(txt)


def _process2(batch_of_batches: List[List[Any]], return_list: List[Any], id: str):
    """Process a batch of batches of data.

    Args:
        batch_of_batches (List[List[Any]]): A batch of batches of data.
        return_list (List[Any]): The list to return the data in.
        id (str): The id of the process.
    """
    for batch in batch_of_batches:
        try:
            return_list.append(COLLATE([DATA[x] for x in batch]))
        except Exception as e:
            if os.path.exists("/mnt/extra/"):
                now = datetime.now()
                with open("/mnt/extra/loader_errs.log", "a") as f:
                    # print( "EXCEPTION TRACE PRINT:\n{}".format( "".join(traceback.format_exception(type(e), e, e.__traceback__))
                    f.write(
                        (
                            f'{now.strftime("%m/%d/%Y, %H:%M:%S")}: {type(e).__name__}: {e}'
                            + "\n"
                        )
                    )

            print("FAILED", id, batch)
            traceback.print_exc(e)


def _collate_none(x: List[Any]) -> Any:
    """Collate function that does nothing.
    
    Args:
        x (List[Any]): The data to collate.
    """
    return x[0]


class MultiLoader(object):

    """Serves up the data for training and testing. An iterator."""

    def __init__(
        self,
        data: Any,
        fragment_representation: str,
        num_dataloader_workers: int = 1,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: function = _collate_none,
        max_voxels_in_memory: int = 80,
    ):
        """Initialize the MultiLoader.

        Args:
            data (Any): The data to load.
            fragment_representation (str): The fragment representation to use.
            num_dataloader_workers (int, optional): The number of dataloader
                workers to use. Defaults to 1.
            batch_size (int, optional): The batch size. Defaults to 1.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to
                False.
            collate_fn (function, optional): The collate function to use. 
                Defaults to _collate_none.
            max_voxels_in_memory (int, optional): The maximum number of voxels
                to keep in memory. Defaults to 80.
        """
        # Save parameters to class variables.
        self.data = data
        self.fragment_representation = fragment_representation
        self.num_dataloader_workers = num_dataloader_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.max_voxels_in_memory = max_voxels_in_memory

        # JDD added below based on
        # https://github.com/pytorch/pytorch/issues/67844 See also
        # https://pytorch.org/docs/stable/multiprocessing.html#multiprocessing-cuda-sharing-details
        multiprocessing.set_sharing_strategy("file_system")

    def __len__(self) -> int:
        """Return number of batches.
        
        Returns:
            int: The number of batches.
        """
        if len(self.data) % self.batch_size == 0:
            return len(self.data) // self.batch_size
        else:
            return (len(self.data) // self.batch_size) + 1

    def _add_procs(self, id: str):
        """Add processors to the pool used to serve up the data. Kill any stale
        ones.
        
        Args:
            id (str): The id of the process.
        """
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
                    print(f"timed out, killing a process: {p.name}")
                    if (
                        "molbert" not in self.fragment_representation
                        and platform.system().lower() != "windows"
                    ):
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
            and len(self.groups_of_batches) > 0
        ):
            batch = self.groups_of_batches.pop(0)

            if (
                "molbert" not in self.fragment_representation
                and platform.system().lower() != "windows"
            ):
                p = multiprocessing.Process(
                    target=_process2,
                    args=(batch, self.return_list, id),
                    name=f"process_{str(self.name_idx + 1)}",
                )
            else:
                p = threading.Thread(
                    target=_process2,
                    args=(batch, self.return_list, id),
                    name=f"process_{str(self.name_idx + 1)}",
                )

            self.name_idx = self.name_idx + 1
            p.start()
            self.procs.append((p, cur_time))

    def __iter__(self):
        """Return an iterator over the batches.
        
        Yields:
            Any: The next batch.
        """
        global DATA
        global COLLATE

        DATA = self.data
        COLLATE = self.collate_fn

        # iden is for debugging. Not critical.
        iden = str(time.time())

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

        # Group the batches. Each of these groups of batches goes to its own
        # processor.
        batches_to_process_per_proc = (
            self.max_voxels_in_memory // self.num_dataloader_workers
        )
        batches_to_process_per_proc = (
            1 if batches_to_process_per_proc == 0 else batches_to_process_per_proc
        )

        self.groups_of_batches = []
        for j in range(1 + len(batches_idxs) // batches_to_process_per_proc):
            group_of_batches = batches_idxs[
                j * batches_to_process_per_proc : (j + 1) * batches_to_process_per_proc
            ]
            if len(group_of_batches) > 0:
                self.groups_of_batches.append(group_of_batches)

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

        num_warnings = 0

        count = 0
        while len(self.groups_of_batches) > 0:
            self._add_procs(iden)

            # Wait until you've got at least one ready
            waited = False
            while len(self.return_list) == 0:
                waited = True
                if num_warnings < 100:
                    print(
                        "Waiting for a voxel grid to finish... If this happens a lot, you might try increasing --max_voxels_in_memory"
                    )
                    num_warnings = num_warnings + 1
                elif num_warnings == 100:
                    print(
                        "Not printing any more warnings about waiting for a voxel grid to finish..."
                    )
                    num_warnings = num_warnings + 1
                time.sleep(0.1)
            if waited:
                print(
                    "Voxel grids finished. Current count: " + str(len(self.return_list))
                )

            # Yield the data as it is needed
            while count < num_data:  # len(self.return_list) > 0 or
                if len(self.return_list) == 0:
                    time.sleep(0.1)
                    continue

                item = self.return_list.pop(0)

                if len(self.return_list) < self.max_voxels_in_memory * 0.1:
                    # Getting low on voxels...
                    self._add_procs(iden)

                count = count + 1
                # print(count)

                # import pdb; pdb.set_trace()

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

    def batch(self, batch_size: int) -> "DataBatch":
        """Batch the data.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            DataBatch: A DataBatch object.
        """
        return DataBatch(self, batch_size)

    def map(self, fn: function) -> "DataLambda":
        """Apply a function to each item in the data.

        Args:
            fn (function): The function to apply.

        Returns:
            DataLambda: A DataLambda object.
        """
        return DataLambda(self, fn)


class DataLambda(MultiLoader):

    """Apply a function to each item in the data."""

    def __init__(self, data: Any, fn: function):
        """Initialize a DataLambda object.
        
        Args:
            data (Any): The data.
            fn (function): The function to apply to each item in the data.
        """
        self.data = data
        self.fn = fn

    def __len__(self) -> int:
        """Get the length of the data.
        
        Returns:
            int: The length of the data.
        """
        return len(self.data)

    def __iter__(self):
        """Iterate over the data.
        
        Yields:
            Any: The data.
        """
        for item in self.data:
            yield self.fn(item)


class DataBatch(MultiLoader):
    
    """Batch the data."""

    def __init__(self, data: Any, batch: int):
        """Initialize a DataBatch object.

        Args:
            data (Any): The data.
            batch (int): The batch size.
        """
        self.data = data
        self.batch = batch

    def __len__(self) -> int:
        """Get the length of the data.

        Returns:
            int: The length of the data.
        """
        if len(self.data) % self.batch == 0:
            return len(self.data) // self.batch
        else:
            return (len(self.data) // self.batch) + 1

    def __iter__(self):
        """Iterate over the data.
        
        Yields:
            Any: The batch of data.
        """
        n = 0
        batch = []
        for item in self.data:
            # if item[3].receptor_name == "Receptor 2v0u":
            #     print(["4", item[3].receptor_name, item[3].fragment_smiles])
            n += 1
            batch.append(item)

            if n == self.batch:
                yield batch
                batch = []
                n = 0

        if n != 0:
            yield batch
