import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple


def run_length_encode(
    arr: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    if len(arr) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    changes = np.concatenate(([True], arr[1:] != arr[:-1]))
    indices = np.where(changes)[0]
    indices = np.concatenate((indices, [len(arr)]))

    values = arr[indices[:-1]]
    counts = indices[1:] - indices[:-1]

    return values, counts
