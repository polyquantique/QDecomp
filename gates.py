from grid_algorithm_1D import Rings as R
import numpy as np

inv_sqrt2 = R.Domega(a = (-1, 1), b = (0, 0), c = (1, 1), d = (0, 0))
_1 = R.Zsqrt2(1, 0)
_0 = R.Zsqrt2(0, 0)
_i = R.Zomega(0, 1, 0, 0)
_omega = R.Zomega(0, 0, 1, 0)


H = np.array([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=R.Domega)

X = np.array([[_0, _1], [_1, _0]], dtype=R.Domega)

Y = np.array([[_0, -_i], [_i, _0]], dtype=R.Domega)

Z = np.array([[_1, _0], [_0, -_1]], dtype=R.Domega)

V = np.array([[R.D(1, 1) * (_1 + _i),  R.D(1, 1) * (_1 - _i)], [R.D(1, 1) * (_1 - _i), R.D(1, 1) * (_1 + _i)]], dtype=R.Domega)

S = np.array([[_1, _0], [_0, _i]], dtype=R.Domega)

T = np.array([[_1, _0], [_0, _omega]], dtype=R.Domega)

print(H@S@H == V)

