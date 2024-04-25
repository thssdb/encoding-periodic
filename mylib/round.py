import numpy as np


def comp_round(dataf, beta):
    dataf = np.copy(dataf)
    ret = []
    for i in range(len(dataf)):
        a = int(round(dataf[i].real / 2**beta))
        b = int(round(dataf[i].imag / 2**beta))
        ret.append(a)
        ret.append(b)
        dataf[i] = a * 2**beta + 1j * b * 2**beta
    return dataf, ret


def comp_round_inverse(ret, beta):
    result = []
    for i in range(0, len(ret), 2):
        result.append(ret[i] * 2**beta + 1j * ret[i + 1] * 2**beta)
    return result
