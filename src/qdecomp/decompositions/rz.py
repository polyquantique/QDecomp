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
The algorithm implemented in this module is based on the algorithm presented by Ross and Selinger in [1]_.

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

.. [1] Neil J. Ross and Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations, https://arxiv.org/pdf/1403.2975.
"""

from qdecomp.utils.exact_synthesis import *
from qdecomp.utils.grid_problem import z_rotational_approximation


def rz_decomp(epsilon: float, angle: float, add_global_phase=False) -> str:
    """
    Decomposes a single-qubit RZ gate its Clifford+T sequence

    Args:
        epsilon (float): The tolerance for the approximation.
        angle (float): The angle of the RZ gate in radians.

    Returns:
        sequence (str): The sequence of Clifford+T gates that approximates the RZ gate.

    """
    # Find the approximation of Rz gates in terms of Domega elements
    Domega_matrix = z_rotational_approximation(epsilon, angle)

    # Convert the Domega matrix to a string representation
    sequence = exact_synthesis_alg(Domega_matrix, print_global_phase=add_global_phase)
    optimized_sequence = optimize_sequence(sequence)

    # Test if TUTdag has less T than U
    tut_sequence = "T " + sequence + "TTTTTTT"
    tut_optimized_sequence = optimize_sequence(tut_sequence)

    # Compare the number of T gates in the two sequences
    if tut_optimized_sequence.count("T") < optimized_sequence.count("T"):
        optimized_sequence = tut_optimized_sequence

    sequence = " ".join(optimized_sequence)
    return sequence


def optimize_sequence(sequence: str) -> str:
    """
    Optimize a sequence of gates by removing redundant gates and combining consecutive gates.

    Args:
        sequence (str): The input sequence of gates as a string.

    Returns:
        str: The optimized sequence of gates.

    Raises:
        TypeError: If the input sequence is not a string.
    """

    if not isinstance(sequence, str):
        raise TypeError(f"Input sequence must be a string. Got {type(sequence)}.")

    # Replace HH by identity

    # Replace TTTT by Z
    optimized_sequence = sequence.replace("TTTT", "Z")

    # Replace TT by S
    optimized_sequence = optimized_sequence.replace("TT", "S")

    optimized_sequence = optimized_sequence.replace("ZZ", "")

    optimized_sequence = optimized_sequence.replace("SSSS", "")

    optimized_sequence = optimized_sequence.replace("HH", "")

    return optimized_sequence
