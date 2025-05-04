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
