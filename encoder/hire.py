import numpy as np
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from encoder import Encoder
from encoder.sprintz import SprintzEncoder

data_global = np.asarray([], dtype=np.int32)
tag_global = []
result_global = []
min_value = []
max_value = []


def compute(id: int, l: int, r: int):
    global data_global, min_value, max_value
    if l == r:
        min_value[id] = data_global[l]
        max_value[id] = data_global[l]
        return
    mid = (l + r) // 2
    compute(id * 2, l, mid)
    compute(id * 2 + 1, mid + 1, r)
    min_value[id] = min(min_value[id * 2], min_value[id * 2 + 1])
    max_value[id] = max(max_value[id * 2], max_value[id * 2 + 1])
    return


def hire(id: int, l: int, r: int, d_value: int):
    global min_value, max_value, tag_global, result_global
    min_cur = min_value[id] - d_value
    max_cur = max_value[id] - d_value
    if min_value == 0 and max_value == 0:
        tag_global.append(0)
        return
    d_cur = (min_cur + max_cur) // 2
    tag_global.append(1)
    result_global.append(d_cur)
    if l == r:
        return
    mid = (l + r) // 2
    hire(id * 2, l, mid, d_value + d_cur)
    hire(id * 2 + 1, mid + 1, r, d_value + d_cur)
    return


class HireDecoder:
    END_CODE = (1 << 31) - 1
    result = []
    tag_global = []
    tag_global_index = 0
    result_global = []
    result_global_index = 0

    def hire(self, l: int, r: int, d_value: int):
        tag_cur = self.tag_global[self.tag_global_index]
        self.tag_global_index += 1
        if tag_cur == 0:
            for i in range(l, r + 1):
                self.result[i] = d_value
                return
        d_cur = self.result_global[self.result_global_index]
        self.result_global_index += 1
        if l == r:
            self.result[l] = (d_value + d_cur) & ((1 << 32) - 1)
            return
        mid = (l + r) // 2
        self.hire(l, mid, (d_value + d_cur) & ((1 << 32) - 1))
        self.hire(mid + 1, r, (d_value + d_cur) & ((1 << 32) - 1))
        return


class HireEncoder(SprintzEncoder):

    decoder = HireDecoder()

    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        global data_global, min_value, max_value, tag_global, result_global
        data_global = np.asarray([], dtype=np.int32)
        tag_global = []
        result_global = []
        min_value = []
        max_value = []
        stream.append(BitArray(uint=len(data), length=32))
        if len(data) != 0:
            data_global = np.asarray(data)
            min_value = [0] * (4 * len(data))
            max_value = [0] * (4 * len(data))
            compute(1, 0, len(data) - 1)
            hire(1, 0, len(data) - 1, 0)
        super().encode_stream(stream, np.asarray(tag_global, dtype=np.int64))
        super().encode_stream(stream, np.asarray(result_global, dtype=np.int64))

    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        self.decoder.result = []
        self.decoder.tag_global = []
        self.decoder.tag_global_index = 0
        self.decoder.result_global = []
        self.decoder.result_global_index = 0
        result_len = int(stream.read(32).uint)
        if result_len == 0:
            return np.asarray([], dtype=np.int64)
        self.decoder.result = [0] * result_len
        self.decoder.tag_global = super().decode(stream).tolist()
        self.decoder.tag_global_index = 0
        self.decoder.result_global = super().decode(stream).tolist()
        self.decoder.result_global_index = 0
        self.decoder.hire(0, result_len - 1, 0)
        for i in range(len(self.decoder.result)):
            if self.decoder.result[i] >= 1 << 31:
                self.decoder.result[i] -= 1 << 32
        return np.asarray(self.decoder.result, dtype=np.int64)

    def get_name(self) -> str:
        return "Hire"
