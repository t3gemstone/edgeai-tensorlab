# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import copy
import itertools
import numpy as np
import torch
from typing import Iterator, Optional, Sized
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class GroupSampler(Sampler):
    """ Implmentation based on InfiniteGroupEachSampleInBatchSampler
    """

    def __init__(self,
                 dataset,
                 shuffle=True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        assert hasattr(dataset, 'flag')

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0

        self.size = len(self.dataset)

        self.batch_size = dataset.batch_size
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = self.batch_size  * world_size
        assert self.groups_num >= self.global_batch_size

        """
        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx)
            for local_sample_idx in range(self.batch_size)]

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        """

        #self.num_samples = 0
        #for i, size in enumerate(self.group_sizes):
        #    self.num_samples += int(
        #        math.ceil(size * 1.0 / self.batch_size /
        #                  self.world_size)) * self.batch_size
        #self.total_size = self.num_samples * self.world_size

    def _infinite_group_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #while True:
            #    yield from torch.randperm(self.groups_num, generator=g).tolist()
            return torch.randperm(self.groups_num, generator=g).tolist()
        else:
            #while True:
            #    yield from torch.arange(self.groups_num).tolist()
            return torch.arange(self.groups_num).tolist()


    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        #yield from itertools.islice(self._infinite_group_indices(),
        return itertools.islice(self._infinite_group_indices(),
                                    global_sample_idx,
                                    None,
                                    self.global_batch_size)

    def __iter__(self) -> Iterator[int]:

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator
        group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx)
            for local_sample_idx in range(self.batch_size)]

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        buffer_per_local_sample = [[] for _ in range(self.batch_size)]

        indices = []
        group_consumed = [0] * self.batch_size
        while True:
            #curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    try:
                        new_group_idx = next(group_indices_per_global_sample_idx[local_sample_idx])
                        buffer_per_local_sample[local_sample_idx] = \
                            copy.deepcopy(
                                group_idx_to_sample_idxs[new_group_idx])
                    except StopIteration:
                        group_consumed[local_sample_idx] = 1
                        continue

                indices.append(buffer_per_local_sample[local_sample_idx].pop(0))

            if all(group_consumed):
                break

        assert len(indices) == self.size
        return iter(indices)
        """
        indices = []
        for i, idx in enumerate(group_indices):
        
            #print(f'Group {i} size: {size}')
            size = self.group_sizes[idx]
            if size == 0:
                continue
            indice = np.where(self.flag == idx)[0].tolist()
            assert len(indice) == size

            extra = int(
                math.ceil(
                    size * 1.0 / self.batch_size / self.world_size)
            ) * self.batch_size * self.world_size - len(indice)

            # pad indice
            tmp = indice.copy()
            for _ in range(extra // size):
                indice.extend(tmp)
            indice.extend(tmp[:extra % size])
            indices.extend(indice)

        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)
        """



    def __len__(self):
        #return self.num_samples
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch