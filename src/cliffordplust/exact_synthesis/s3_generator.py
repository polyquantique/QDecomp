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
This module provides two functions for generating all entries in the S3 table of the Clifford+T gate set.  
The first one generates a list of strings containing all the possible sequences of T and H gates with at  
most 7 consecutive T gates, at most 3 H gates and starting with an H gate. The second function generates  
the first column of the matrix given by each string of the sequence produced by the first function and  
stores it in a json file.
"""

import json
import os


from cliffordplust.exact_synthesis import apply_sequence, convert_to_tuple


def generate_sequences(max_consecutive_t: int = 7) -> list[str]:
    """Generate all valid sequences of T and H gates with a maximum number of consecutive T gates.

    Args:
        max_consecutive_t (int): Maximum number of consecutive T gates

    Returns:
        list: List of valid sequences of T and H gates in string format
    """
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
                            valid_sequences.append(
                                "H" + "T" * n_3 + "H" + "T" * n_2 + "H"
                            )
                        else:
                            valid_sequences.append(
                                "T" * n_3 + "H" + "T" * n_2 + "H" + "T" * n_1 + "H"
                            )
    return valid_sequences


def generate_s3(max_consecutive_t: int = 7) -> None:
    """Generate the S3 table for the first column of Clifford+ gate set and write it to a JSON file.
    Args:
        max_consecutive_t (int): Maximum number of consecutive T gates
    """
    s3_sequences = generate_sequences(max_consecutive_t)
    s3_dict = {seq: convert_to_tuple(apply_sequence(seq)) for seq in s3_sequences}

    def serialize_dict(d):
        return {
            k: [[list(inner) for inner in outer] for outer in v] for k, v in d.items()
        }

    with open(os.path.join(os.path.dirname(__file__), "s3_table.json"), "w") as f:
        json.dump(serialize_dict(s3_dict), f, indent=4)
