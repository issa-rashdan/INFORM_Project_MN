import itertools
from typing import List, Iterable, Tuple, Generator, Callable
import torch
from torch.utils.data import Dataset
import numpy as np

class GroupedGenerator:
    def __init__(self, parts: list[Callable[[], Iterable]]):
        self.parts = parts

    def __call__(self):
        return itertools.chain.from_iterable(part() for part in self.parts)


class BatchGeneratorDataset(Dataset):
    def __init__(
        self,
        grouped_generator_fns: List[Callable[[], Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    ):
        # each entry is a callable that returns (label_patches, data_patches, mask_patches)
        self.grouped_generator_fns = grouped_generator_fns

    def __len__(self) -> int:
        return len(self.grouped_generator_fns)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_labels, batch_data, batch_masks = [], [], []
        # call the idx-th generator fn to get fresh patch-stream
        for label_np, data_np, mask_np in self.grouped_generator_fns[idx]():
            batch_labels.append(label_np)
            batch_data.append(data_np)
            batch_masks.append(mask_np)

        # concatenate along the batch axis (axis=0)
        lbls = np.concatenate(batch_labels, axis=0)
        dts  = np.concatenate(batch_data,   axis=0)
        msks = np.concatenate(batch_masks,  axis=0)

        # convert once to torch.Tensor
        # assume labels are integer classes, data and masks are floats
        lbls = torch.from_numpy(lbls).long()
        dts  = torch.from_numpy(dts).float()
        msks = torch.from_numpy(msks).bool()

        return lbls, dts, msks



def group_generators_by_patch_limit(
    generator_fns: List[Callable[[], Generator]],  # NOTE: changed type
    patch_counts: List[int],
    batch_limit: int
):
    assert len(generator_fns) == len(patch_counts)

    grouped_generator_fns = []
    grouped_patch_counts = []
    current_group = []
    current_group_counts = []
    current_count = 0

    for gen_fn, count in zip(generator_fns, patch_counts):
        if current_count + count <= batch_limit:
            current_group.append(gen_fn)
            current_group_counts.append(count)
            current_count += count
        else:
            # wrap the slice of gen_fns in a picklable class
            grouped_generator_fns.append(GroupedGenerator(current_group))
            grouped_patch_counts.append(current_group)
            current_group = [gen_fn]; current_group_counts = [count]; current_count = count

            current_group = [gen_fn]
            current_group_counts = [count]
            current_count = count

    if current_group:
        grouped_generator_fns.append(GroupedGenerator(current_group))
        grouped_patch_counts.append(current_group_counts)

    return grouped_generator_fns, grouped_patch_counts



