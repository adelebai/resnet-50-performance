'''
Dataloader worker code adapted from: https://github.com/pytorch/pytorch/blob/fbdb86c1747737c744ad79b5da6bcbd064dc982e/torch/utils/data/_utils/worker.py

The `custom_worker` loop will be monkey-patched in at runtime to replace original implementation.
'''
import torch
import random
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from torch._utils import ExceptionWrapper
from typing import Optional, Union, TYPE_CHECKING
from torch.utils.data import _DatasetKind
from torch.utils.data._utils import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS, HAS_NUMPY
from torch.utils.data._utils.worker import _generate_state, ManagerWatchdog, WorkerInfo, _IterableDatasetStopIteration, _IterableDatasetStopIteration, _ResumeIteration
if TYPE_CHECKING:
    from torch.utils.data import Dataset


iteration_end = False

def custom_fetch(fetcher, index, dataset_kind, worker_id, data_queue, idx, r):
    try:
        data = fetcher.fetch(index)
    except Exception as e:
        if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
            data = _IterableDatasetStopIteration(worker_id)
            # Set `iteration_end`
            #   (1) to save future `next(...)` calls, and
            #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
            
            # sleep a bit since this should be queued last
            time.sleep(5)
            iteration_end = True
        else:
            # It is important that we don't store exc_info in a variable.
            # `ExceptionWrapper` does the correct thing.
            # See NOTE [ Python Traceback Reference Cycle Problem ]
            data = ExceptionWrapper(
                where="in DataLoader worker process {}".format(worker_id))
    data_queue.put((idx, data))
    del data, idx, index, r  # save memory

def custom_worker(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                 num_workers, persistent_workers, shared_seed):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        if HAS_NUMPY:
            np_seed = _generate_state(base_seed, worker_id)
            import numpy as np
            np.random.seed(np_seed)

        from torch.utils.data import IterDataPipe
        from torch.utils.data.graph_settings import apply_random_seed

        shared_rng = torch.Generator()
        if isinstance(dataset, IterDataPipe):
            assert shared_seed is not None
            shared_rng.manual_seed(shared_seed)
            dataset = apply_random_seed(dataset, shared_rng)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers,
                                  seed=seed, dataset=dataset)

        

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(
                where="in DataLoader worker process {}".format(worker_id))

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()
        executor = ThreadPoolExecutor(os.cpu_count())

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                executor.shutdown(wait=True)
                executor = ThreadPoolExecutor(os.cpu_count())
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    assert r.seed is not None
                    shared_rng.manual_seed(r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                continue
            elif r is None:
                executor.shutdown(wait=True)
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, index = r
            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            if init_exception is not None:
                data = init_exception
                init_exception = None
                data_queue.put((idx, data))
                del data, idx, index, r  # save memory
            else:
                executor.submit(custom_fetch, fetcher, index, dataset_kind, worker_id, data_queue, idx, r)

    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    executor.shutdown(wait=True)
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()