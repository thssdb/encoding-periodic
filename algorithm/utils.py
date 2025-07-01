import numpy as np
from numpy.typing import NDArray


def get_bit_length_cnt(data: NDArray[np.uint64]) -> NDArray[np.int64]:
    n = data.shape[0]
    cnt = np.zeros(int(np.max(data)).bit_length() + 1, dtype=np.int64)
    for i in range(n):
        cnt[int(data[i]).bit_length()] += 1
    return cnt
