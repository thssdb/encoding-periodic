import numpy as np
import pandas as pd
import time
from numpy.typing import NDArray
from bitstring import BitStream, BitArray
from algorithm.constants import MAX_LENGTH, MAX_VALUE
from abc import ABC, abstractmethod


class Encoder(ABC):
    @abstractmethod
    def encode_stream(self, stream: BitStream, data: NDArray[np.int64]):
        pass

    def encode(self, data: NDArray[np.int64]) -> BitStream:
        stream = BitStream()
        self.encode_stream(stream, data)
        return stream

    @abstractmethod
    def decode(self, stream: BitStream) -> NDArray[np.int64]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def exp(self, data: NDArray[np.int64]) -> pd.DataFrame:
        start_time = time.time()
        stream = self.encode(data)
        encoding_time = time.time() - start_time

        stream.pos = 0

        start_time = time.time()
        decoded_data = self.decode(stream)
        decoding_time = time.time() - start_time

        assert np.array_equal(
            data, decoded_data
        ), "Decoded data does not match original data."

        return pd.DataFrame(
            {
                "encoder": [self.get_name()],
                "encoding_time": [encoding_time],
                "decoding_time": [decoding_time],
                "data_size": [data.nbytes],
                "stream_size": [stream.length // 8],
            }
        )
