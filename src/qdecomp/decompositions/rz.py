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
This module contains the main function for the decomposition of :math:`R_z` gates of the form

.. math:: 
    
    R_z = \\begin{pmatrix}
    e^{-i\\theta / 2} & 0  \\\\
    0 & e^{i\\theta / 2}
    \\end{pmatrix},

where :math:`\\theta` is the rotation angle around the Z axis. The :math:`R_z` gate is decomposed into a sequence of Clifford+T gates up to a given :math:`\\varepsilon` tolerance.
The algorithm implemented in this module is based on the algorithm presented by Ross and Selinger in [#Ross]_. Note: when the `add_global_phase` argument is set to `True`, the sequence
will include global phase gates :math:`W = e^{i\\pi/4}`.

This module combines the :mod:`qdecomp.utils.exact_synthesis`, :mod:`qdecomp.utils.grid_problem` and :mod:`qdecomp.utils.diophantine` modules to achieve this goal.

 **Example**

    .. code-block:: python

        >>> from qdecomp.decompositions import rz_decomp
        >>> from math import pi
        
        # Decompose a RZ gate with angle pi/128 and tolerance 0.001 exactly
        >>> sequence = rz_decomp(epsilon=0.001, angle=pi/128, add_global_phase=True)
        >>> print(sequence)
        H S T H S T H [...] Z S W W W W W W

        # Decompose a RZ gate with angle pi/128 and tolerance 0.001 up to a global phase
        >>> sequence = rz_decomp(epsilon=0.001, angle=pi/128, add_global_phase=False)
        >>> print(sequence)
        H S T H S T H [...] Z S H S T H Z S

.. [#Ross] Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, https://arxiv.org/pdf/1403.2975.
"""

from qdecomp.utils.exact_synthesis import exact_synthesis_alg, optimize_sequence
from qdecomp.utils.grid_problem import z_rotational_approximation


def rz_decomp(angle: float, epsilon: float, add_global_phase=False) -> str:
    """
    Decomposes a single-qubit RZ gate its Clifford+T sequence

    Args:
        angle (float): The angle of the RZ gate in radians.
        epsilon (float): The tolerance for the approximation based on the operator norm.
        add_global_phase (bool): If `True`, adds global phase gates W to the sequence (default: `False`).

    Returns:
        sequence (str): The sequence of Clifford+T gates that approximates the RZ gate.

    """
    # Find the approximation of Rz gates in terms of Domega elements
    domega_matrix = z_rotational_approximation(epsilon=epsilon, theta=angle)

    # Convert the Domega matrix to a string representation
    sequence = exact_synthesis_alg(domega_matrix, insert_global_phase=add_global_phase)
    optimized_sequence = optimize_sequence(sequence)

    # Test if TUTdag has less T than U
    tut_sequence = "T" + sequence + "TTTTTTT"
    tut_optimized_sequence = optimize_sequence(tut_sequence)

    # Compare the number of T gates in the two sequences
    if tut_optimized_sequence.count("T") < optimized_sequence.count("T"):
        optimized_sequence = tut_optimized_sequence

    sequence = " ".join(optimized_sequence)
    return sequence
