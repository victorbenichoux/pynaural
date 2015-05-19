from scipy.io import loadmat, savemat
import numpy as np


def append_at_root(m0, m1):
    """
    Appends m1 to the m0 mat file. The new mat file has two roots, m0 and m1.

    :param m0:
    :param m1:
    :return:
    """

    fm0 = loadmat(m0)
    fm1 = loadmat(m1)

    dtype = [('m0', object), ('m1', object)]
    out = np.zeros((1,1), dtype = dtype)

    out['m0'][0,0] = fm0
    out['m1'][0,0] = fm1

    savemat()

def append_in_mem(m, )