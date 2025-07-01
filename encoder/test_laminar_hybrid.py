import numpy as np
from encoder.laminar_hybrid import LaminarHybridEncoder


def get_dense_mode(real_bit_width: int, group_bit_width: int, beta: int) -> int:
    if beta < group_bit_width:
        return 1 + group_bit_width - beta
    return 0


def get_sparse_mode(
    real_bit_width: int, group_bit_width: int, beta: int, index_bit_length: int
) -> int:
    if beta < real_bit_width:
        return 1 + group_bit_width - beta + index_bit_length
    return 0


def test_add_dense_mode():
    min_beta = 5
    max_beta = 20
    encoder = LaminarHybridEncoder(min_beta=min_beta, max_beta=max_beta)

    bit_widths = ((10, 12), (0, 3), (5, 16), (15, 20), (26, 29), (7, 7))
    for real_bit_width, group_bit_width in bit_widths:
        result_d2 = np.zeros(max_beta - min_beta + 1, dtype=np.int64)
        encoder.add_dense_mode(result_d2, real_bit_width, group_bit_width)
        result = encoder.calculate_result(result_d2)
        expected_result = np.zeros(max_beta - min_beta + 1, dtype=np.int64)
        for beta in range(min_beta, max_beta + 1):
            expected_result[beta - min_beta] = get_dense_mode(
                real_bit_width, group_bit_width, beta
            )
        assert np.array_equal(
            result, expected_result
        ), f"Expected {expected_result}, got {result}"


def test_add_sparse_mode():
    min_beta = 5
    max_beta = 20
    encoder = LaminarHybridEncoder(min_beta=min_beta, max_beta=max_beta)

    bit_widths = ((10, 12), (0, 3), (5, 16), (15, 20), (26, 29), (7, 7))
    for real_bit_width, group_bit_width in bit_widths:
        index_bit_length = (real_bit_width - 1).bit_length()
        result_d2 = np.zeros(max_beta - min_beta + 1, dtype=np.int64)
        encoder.add_sparse_mode(
            result_d2, real_bit_width, group_bit_width, index_bit_length
        )
        result = encoder.calculate_result(result_d2)
        expected_result = np.zeros(max_beta - min_beta + 1, dtype=np.int64)
        for beta in range(min_beta, max_beta + 1):
            expected_result[beta - min_beta] = get_sparse_mode(
                real_bit_width, group_bit_width, beta, index_bit_length
            )
        assert np.array_equal(
            result, expected_result
        ), f"Expected {expected_result}, got {result}"


def test_laminar_hybrid_encoder():
    encoder = LaminarHybridEncoder()
    data = np.array(
        [100, -50, 9, -82, 3, -8, 7, 10, 10, -5, 0, 2, 0, 1, 0, 0, 0], dtype=np.int64
    )
    result = encoder.exp(data)
    assert (result["encoder"].to_numpy())[0] == "Laminar-Hybrid"
