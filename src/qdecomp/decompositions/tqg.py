# Copyright 2024-2025 Olivier Romain, Francis Blais, Vincent Girouard, Marius Trudeau
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Union
import numpy as np

from qdecomp.decompositions import sqg_decomp, cnot_decomposition
from qdecomp.utils import QGate


def tqg_decomp(tqg: Union[np.array, QGate], epsilon: float = 0.01) -> list[QGate]:
    """
    Decomposes a two-qubit gate (TQG) into its sequence of CNOT and single qubit gates.

    Args:
        tqg (Union[np.array, QGate]): The two-qubit gate to decompose.
        epsilon (float): The tolerance for the decomposition (default: 0.01).

    Returns:
        list[QGate]: A list of QGate objects representing the decomposed gates with their sequences defined
    """

    if type(tqg) is not np.ndarray and type(tqg) is not QGate:
        raise ValueError("Input must be a numpy array or QGate object, got " + str(type(tqg)))

    if type(tqg) is not QGate:
        tqg_matrix = tqg
        tqg = QGate.from_matrix(matrix=tqg_matrix, matrix_target=(0, 1), epsilon=epsilon)

    if tqg.matrix.shape != (4, 4):
        raise ValueError("Input gate must be a 4x4 matrix, got " + str(tqg.matrix.shape))

    cnot_decomp_lists = cnot_decomposition(tqg.matrix)

    # Decompose each gate in the cnot decomposition list
    for cnot_decomp_qgate in cnot_decomp_lists:
        # if gate sequence is already initialized, skip decomposition
        if cnot_decomp_qgate.sequence is not None:
            continue

        # Else, perform the sqg decomposition
        else:
            cnot_qgate_seq, alpha = sqg_decomp(cnot_decomp_qgate.matrix, epsilon=epsilon)
            cnot_decomp_qgate.set_decomposition(cnot_qgate_seq, epsilon=epsilon)

        # Decomposition of all single qubit gates using zyz, rz_approx and exact synthesis
    return cnot_decomp_lists
