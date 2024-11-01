from Dsqrt2 import Dsqrt2
from CliffordPlusT.Domega import D
import numpy as np

def R():
    return np.matrix([[Dsqrt2(0, D(1, 1)),-Dsqrt2(0, D(1, 1))],[Dsqrt2(0, D(1, 1)),Dsqrt2(0, D(1, 1))]])