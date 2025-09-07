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
This module generates the S3 table of all Clifford+T gates with sde :math:`\\leq 3` and provides a utility function
to help generate the table.

This module contains the following functions:

- :func:`generate_sequences`: Generates all valid sequences of T and H gates with specific constraints (see function description).
- :func:`generate_s3`: Generates the S3 table of Clifford+T gates with sde :math:`\\leq 3` and stores it in a JSON file.
"""

import json
import os

from qdecomp.rings import Domega
from qdecomp.utils.exact_synthesis.exact_synthesis import apply_sequence, domega_matrix_to_tuple


def generate_sequences() -> list[str]:
    """
    Generate all valid sequences of T and H gates.

    The sequences that are viable candidates in the S3 table are generated with the following constraints:

    - A maximum of 7 consecutive T gates.
    - A maximum of 3 H gates.
    - The sequence starts with an H gate.

    Returns:
        list[str]: A list of strings containing the valid sequences of T and H gates.
    """
    max_consecutive_t = 7
    valid_sequences = []

    for n_3 in range(0, max_consecutive_t + 1):
        if n_3 == 0:
            valid_sequences.append("H")
            valid_sequences.append("")

        else:
            for n_2 in range(0, max_consecutive_t + 1):
                if n_2 == 0:
                    valid_sequences.append("T" * n_3 + "H")
                    valid_sequences.append("H" + "T" * n_3 + "H")

                else:
                    for n_1 in range(0, max_consecutive_t + 1):
                        if n_1 == 0:
                            valid_sequences.append("T" * n_3 + "H" + "T" * n_2 + "H")
                            valid_sequences.append("H" + "T" * n_3 + "H" + "T" * n_2 + "H")

                        else:
                            valid_sequences.append(
                                "T" * n_3 + "H" + "T" * n_2 + "H" + "T" * n_1 + "H"
                            )

    return valid_sequences


def generate_s3() -> None:
    """
    Generate the S3 table of all Clifford+T gates with sde :math:`\\leq 3` up to a global phase.

    This function generates the first column of the matrix given by each string
    of the sequence produced by the :func:`generate_sequences` function. Each element from the
    matrix is stored as a tuple with eight integers required to initialize a Domega object.
    It stores the result in a JSON file named `s3_table.json` in the same directory as
    this script.
    """
    s3_sequences = generate_sequences()
    s3_dict = {seq: domega_matrix_to_tuple(apply_sequence(seq)) for seq in s3_sequences}

    # Remove duplicate values in s3_dict
    unique_values: dict = {}
    for key, value in s3_dict.items():
        # Check if the sde is greater than 3, if so do not add to table
        if (
            Domega(value[0][0], value[0][1], value[0][2], value[0][3])
            * Domega(value[0][0], value[0][1], value[0][2], value[0][3]).complex_conjugate()
        ).sde() > 3:
            continue

        if value not in unique_values.values():
            unique_values[key] = value
        else:
            # Replace the existing key if the current key has fewer 'T' gates
            existing_key = next(k for k, v in unique_values.items() if v == value)
            if key.count("T") < existing_key.count("T"):
                del unique_values[existing_key]
                unique_values[key] = value

    s3_dict = unique_values
    # Serialize the S3 dictionary
    serialized_dict = json.dumps(s3_dict).replace(', "', ',\n"')

    # Save the dictionary
    with open(os.path.join(os.path.dirname(__file__), "s3_table.json"), "w") as f:
        f.write(serialized_dict)
