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

"""
This module contains the main function for the decomposition of two qubit gates (TQG) into a sequence of Clifford+T gates up to a given tolerance :math:`\\varepsilon`.
This module uses and combines the :mod:`qdecomp.decompositions.sqg` and :mod:`qdecomp.decompositions.cnot` decomposition algorithms to achieve this goal.

**Example**

    .. code-block:: python

        >>> from scipy.stats import unitary_group
        >>> from qdecomp.decompositions import sqg_decomp

        # Decompose a random two qubit gate with tolerance 0.001
        >>> tqg = unitary_group.rvs(4, random_state=42)
        >>> circuit = tqg_decomp(tqg, epsilon=0.001)
        >>> for gates in circuit:
        >>>     print(f"{gates.target} -> {gates.sequence}")
        (0,) -> S T H T [...] H Z S T
        (1,) -> S T H T [...] S H S T
        (0, 1) -> CNOT1
        ...
        (1,) -> H T H S [...] T H Z S

"""

from typing import Union
import numpy as np

from qdecomp.decompositions import sqg_decomp, cnot_decomposition
from qdecomp.utils import QGate


def tqg_decomp(tqg: Union[np.ndarray, QGate], epsilon: float = 0.01) -> list[QGate]:
    """
    Decomposes a two-qubit gate (TQG) into its optimal sequence of CNOT and single qubit gates.

    Args:
        tqg (Union[np.array, QGate]): The two-qubit gate to decompose.
        epsilon (float): The tolerance for the decomposition (default: 0.01).

    Returns:
        list[QGate]: A list of QGate objects representing the decomposed gates with their sequences defined

    Raises:
        TypeError: If the input is not a numpy array or QGate object.
        ValueError: If the input is not a 4x4 matrix or QGate object with a 4x4 matrix.
    """

    if not isinstance(tqg, (np.ndarray, QGate)):
        raise TypeError("Input must be a numpy array or QGate object, got " + str(type(tqg)))

    if not isinstance(tqg, QGate):
        tqg_matrix = tqg
        if tqg_matrix.shape != (4, 4):
            raise ValueError("Input gate must be a 4x4 matrix, got " + str(tqg.shape))
        tqg = QGate.from_matrix(matrix=tqg_matrix, target=(0, 1), epsilon=epsilon)

    if tqg.init_matrix.shape != (4, 4):
        raise ValueError("Input gate must be a 4x4 matrix, got " + str(tqg.init_matrix.shape))

    cnot_decomp_lists = cnot_decomposition(tqg.init_matrix)

    # Decompose each gate in the cnot decomposition list
    for cnot_decomp_qgate in cnot_decomp_lists:
        # if gate sequence is already initialized, skip decomposition
        if cnot_decomp_qgate.sequence is not None:
            continue

        # Else, perform the sqg decomposition
        else:
            cnot_qgate_seq, alpha = sqg_decomp(cnot_decomp_qgate.init_matrix, epsilon=epsilon)
            cnot_decomp_qgate.set_decomposition(cnot_qgate_seq, epsilon=epsilon)

    return cnot_decomp_lists
